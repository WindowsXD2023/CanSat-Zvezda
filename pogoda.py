"""Program pobiera dwudniową prognozę i rysuje jej przebieg na różnych wysokościach.

Kolor tła oznacza temperaturę, strzałki pokazują wiatr, a linie ciśnienie.
Kliknięcie wykresu wyświetla szczegóły najbliższego punktu prognozy.
"""

# requests wysyła zapytania do serwisów internetowych.
import requests
# pandas ułatwia zamianę tekstowej daty z API na datę rozumianą przez program.
import pandas as pd
# numpy służy do obliczeń na dużych zestawach liczb, czyli tablicach.
import numpy as np

# matplotlib rysuje wykresy. TkAgg otwiera wykres w osobnym oknie systemowym.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# DateFormatter ustawia czytelny sposób wyświetlania dat na osi czasu.
from matplotlib.dates import DateFormatter
# Widgety Matplotlib dodają do wykresu pola tekstowe, przyciski i wybór modelu.
from matplotlib.widgets import Button, RadioButtons, TextBox


def pobierz_profil_wysokosciowy(szerokosc_geograficzna, dlugosc_geograficzna, model=None):
    """Pobiera prognozę dla wybranych współrzędnych i modelu."""
    # Poziomy ciśnienia atmosferycznego co 25 hPa; każdy poziom dostarcza dane z innej wysokości.
    poziomy_hpa = list(range(975, 724, -25))

    # Przygotowujemy nazwy danych, o które poprosimy serwis pogodowy.
    zmienne_api = []
    for poziom in poziomy_hpa:
        # Dla każdego poziomu chcemy temperaturę, wiatr, jego kierunek i wysokość poziomu.
        zmienne_api.extend([
            f"temperature_{poziom}hPa",
            f"wind_speed_{poziom}hPa",
            f"wind_direction_{poziom}hPa",
            f"geopotential_height_{poziom}hPa",
        ])

    # Dodajemy też dane blisko powierzchni, aby oszacować warunki przy ziemi.
    zmienne_api.extend([
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
    ])

    # Adres internetowy usługi zwracającej prognozę pogody.
    url = "https://api.open-meteo.com/v1/forecast"

    # Parametry zapytania: miejsce, żądane pola, strefa czasowa, długość prognozy i jednostka wiatru.
    params = {
        "latitude": szerokosc_geograficzna,
        "longitude": dlugosc_geograficzna,
        "hourly": ",".join(zmienne_api),
        "wind_speed_unit": "ms",
        "timezone": "Europe/Warsaw",
        "forecast_days": 2,
    }
    # Gdy model jest ustawiony na None, Open-Meteo automatycznie dobiera prognozę.
    # W przeciwnym razie prosimy serwis o konkretny model, np. GFS albo ICON.
    if model is not None:
        params["models"] = model

    # Wysyłamy zapytanie. Limit czasu zapobiega czekaniu bez końca na odpowiedź.
    odpowiedz = requests.get(url, params=params, timeout=30)
    # Jeśli serwer zwrócił błąd, ta instrukcja pokaże go zamiast używać błędnej odpowiedzi.
    odpowiedz.raise_for_status()
    # Odpowiedź JSON to uporządkowane dane, które zamieniamy na słownik Pythona.
    dane = odpowiedz.json()
    # Niektóre poziomy ciśnienia mogą leżeć pod gruntem i wtedy API zwraca dla nich None.
    # Sprawdzamy również, czy dane sięgają do górnej granicy wykresu (2500 m nad gruntem).
    liczba_poziomow_z_danymi = sum(
        any(wartosc is not None for wartosc in dane["hourly"].get(f"temperature_{poziom}hPa", []))
        for poziom in poziomy_hpa
    )
    if liczba_poziomow_z_danymi < 2:
        raise ValueError("Wybrany model nie zwrócił danych wysokościowych dla tej lokalizacji.")
    najwyzsza_wysokosc_w_godzinie = np.full(len(dane["hourly"]["time"]), -np.inf)
    for poziom in poziomy_hpa:
        wysokosci = np.asarray(
            dane["hourly"].get(f"geopotential_height_{poziom}hPa", []), dtype=float
        ) - dane["elevation"]
        poprawne_wysokosci = np.isfinite(wysokosci)
        najwyzsza_wysokosc_w_godzinie[poprawne_wysokosci] = np.maximum(
            najwyzsza_wysokosc_w_godzinie[poprawne_wysokosci],
            wysokosci[poprawne_wysokosci],
        )
    if np.any(najwyzsza_wysokosc_w_godzinie < 2500):
        raise ValueError(
            "Ten model nie ma danych do 2500 m nad gruntem w tej lokalizacji. "
            "Wybierz Automatyczny [ZALECANY] albo GFS."
        )

    # Zwracamy prognozę godzinową, wysokość terenu nad poziomem morza i listę poziomów hPa.
    return dane["hourly"], dane["elevation"], poziomy_hpa


def interpoluj_na_wysokosci(wysokosci_zrodlowe, wartosci_zrodlowe, wysokosci_docelowe):
    """Szacuje wartości na wybranych wysokościach pomiędzy poziomami z prognozy."""
    # Tworzymy pustą tablicę wynikową: wiersze to wysokości, kolumny to godziny.
    # Na początku wpisujemy NaN, czyli specjalną wartość oznaczającą brak liczby.
    wynik = np.full((len(wysokosci_docelowe), wartosci_zrodlowe.shape[1]), np.nan)

    # Wykonujemy obliczenie osobno dla każdej godziny prognozy.
    for indeks_czasu in range(wartosci_zrodlowe.shape[1]):
        # Wybieramy wszystkie wysokości i wartości dla aktualnej godziny.
        wysokosci = wysokosci_zrodlowe[:, indeks_czasu]
        wartosci = wartosci_zrodlowe[:, indeks_czasu]

        # Pomijamy brakujące dane; do interpolacji potrzebujemy co najmniej dwóch punktów.
        poprawne = np.isfinite(wysokosci) & np.isfinite(wartosci)
        if np.count_nonzero(poprawne) < 2:
            continue

        # np.interp wymaga wysokości w kolejności rosnącej, dlatego porządkujemy punkty.
        kolejnosc = np.argsort(wysokosci[poprawne])
        wynik[:, indeks_czasu] = np.interp(
            wysokosci_docelowe,
            wysokosci[poprawne][kolejnosc],
            wartosci[poprawne][kolejnosc],
        )

    # Zwracamy tablicę z oszacowanymi wartościami dla każdej wysokości i godziny.
    return wynik


def przygotuj_dane_wysokosciowe(dane_api, elewacja_m, poziomy_hpa):
    """Układa dane z API w tablice dla wysokości od 0 do 2500 metrów."""
    # Liczba godzin wynika z liczby dat zwróconych przez serwis.
    liczba_godzin = len(dane_api["time"])
    # Wysokości wykresu: od 0 do 2500 m, co 100 m (łącznie 26 poziomów).
    wysokosci_docelowe = np.arange(0, 2501, 100)

    # API podaje wysokość poziomu ciśnienia nad poziomem morza.
    # Odejmujemy wysokość terenu, aby otrzymać wysokość nad lokalnym gruntem.
    wysokosci_poziomow = np.vstack([
        np.asarray(dane_api[f"geopotential_height_{poziom}hPa"], dtype=float) - elewacja_m
        for poziom in poziomy_hpa
    ])

    # Do profilu dokładamy pomiar wiatru na 10 m jako najniższy punkt odniesienia.
    wysokosci_zrodlowe = np.vstack([
        np.full(liczba_godzin, 10.0),
        wysokosci_poziomow,
    ])

    # Budujemy tablicę temperatury: pierwszy wiersz to pomiar przy powierzchni,
    # a kolejne wiersze to dane z poszczególnych poziomów ciśnienia.
    temperatura_zrodlowa = np.vstack([
        np.asarray(dane_api["temperature_2m"], dtype=float),
        *[
            np.asarray(dane_api[f"temperature_{poziom}hPa"], dtype=float)
            for poziom in poziomy_hpa
        ],
    ])

    # Prędkość wiatru jest już pobierana w metrach na sekundę (m/s).
    wiatr_speed_zrodlowy = np.vstack([
        np.asarray(dane_api["wind_speed_10m"], dtype=float),
        *[
            np.asarray(dane_api[f"wind_speed_{poziom}hPa"], dtype=float)
            for poziom in poziomy_hpa
        ],
    ])

    # Kierunek wiatru podawany jest w stopniach, zgodnie z kompasem.
    wiatr_kierunek_zrodlowy = np.vstack([
        np.asarray(dane_api["wind_direction_10m"], dtype=float),
        *[
            np.asarray(dane_api[f"wind_direction_{poziom}hPa"], dtype=float)
            for poziom in poziomy_hpa
        ],
    ])

    # Dla powierzchni API podaje ciśnienie w danej godzinie; wyżej używamy
    # wartości poziomu, np. 900 hPa oznacza poziom o ciśnieniu 900 hPa.
    cisnienie_zrodlowe = np.vstack([
        np.asarray(dane_api["surface_pressure"], dtype=float),
        *[
            np.full(liczba_godzin, poziom, dtype=float)
            for poziom in poziomy_hpa
        ],
    ])

    # Zamieniamy kąt na radiany, bo funkcje sinus i cosinus używają tej jednostki.
    wiatr_kierunek_rad = np.deg2rad(wiatr_kierunek_zrodlowy)
    # Wiatr rysujemy jako dwie składowe: poziomą (wschód-zachód) i pionową (północ-południe).
    wiatr_u_zrodlowy = -wiatr_speed_zrodlowy * np.sin(wiatr_kierunek_rad)
    wiatr_v_zrodlowy = -wiatr_speed_zrodlowy * np.cos(wiatr_kierunek_rad)

    # Szacujemy pogodę dla regularnej siatki co 100 m między poziomami z API.
    temperatura = interpoluj_na_wysokosci(
        wysokosci_zrodlowe, temperatura_zrodlowa, wysokosci_docelowe
    )
    wiatr_u = interpoluj_na_wysokosci(
        wysokosci_zrodlowe, wiatr_u_zrodlowy, wysokosci_docelowe
    )
    wiatr_v = interpoluj_na_wysokosci(
        wysokosci_zrodlowe, wiatr_v_zrodlowy, wysokosci_docelowe
    )
    cisnienie = interpoluj_na_wysokosci(
        wysokosci_zrodlowe, cisnienie_zrodlowe, wysokosci_docelowe
    )

    # Odtwarzamy prędkość i kierunek z dwóch składowych wiatru.
    wiatr_speed = np.hypot(wiatr_u, wiatr_v)
    wiatr_kierunek = np.mod(np.rad2deg(np.arctan2(-wiatr_u, -wiatr_v)), 360)

    # Słownik przechowuje serie czasowe pod czytelnymi nazwami z jednostką i wysokością.
    # Przykładowo klucz "wiatr_speed_m_s_100m" zawiera prędkość wiatru na 100 m
    # dla wszystkich 48 godzin prognozy, a nie tylko dla jednej chwili.
    zmienne_dynamiczne = {}
    for indeks, wysokosc_m in enumerate(wysokosci_docelowe):
        zmienne_dynamiczne[f"temperatura_C_{wysokosc_m}m"] = temperatura[indeks]
        zmienne_dynamiczne[f"wiatr_speed_m_s_{wysokosc_m}m"] = wiatr_speed[indeks]
        zmienne_dynamiczne[f"wiatr_kierunek_stopnie_{wysokosc_m}m"] = wiatr_kierunek[indeks]
        zmienne_dynamiczne[f"cisnienie_hPa_{wysokosc_m}m"] = cisnienie[indeks]

    # Zwracamy wysokości, tablice wykresu i słownik wszystkich serii czasowych.
    return wysokosci_docelowe, temperatura, wiatr_u, wiatr_v, cisnienie, zmienne_dynamiczne


def glowna_aplikacja(szerokosc_startowa=50.34, dlugosc_startowa=19.51):
    """Pokazuje wykres oraz panel do zmiany modelu i współrzędnych."""
    # Te etykiety widzi użytkownik; wartości po prawej stronie to nazwy modeli API.
    modele = {
        "Automatyczny [ZALECANY dla CanSat]": None,
        "GFS Seamless (porównanie)": "gfs_seamless",
    }

    # Tworzymy jedno okno, które będzie odświeżane po każdej zmianie ustawień.
    fig = plt.figure(figsize=(15, 8))
    ostatnie_zmienne = {}

    def rysuj_prognoze(szerokosc_geograficzna, dlugosc_geograficzna, model):
        """Pobiera wybrany wariant prognozy i odtwarza zawartość okna."""
        nonlocal ostatnie_zmienne

        # Najpierw pobieramy i przygotowujemy nowe dane; stary wykres zostaje,
        # jeśli pobieranie się nie powiedzie.
        dane_api, elewacja_m, poziomy_hpa = pobierz_profil_wysokosciowy(
            szerokosc_geograficzna, dlugosc_geograficzna, model
        )
        czasy = pd.to_datetime(dane_api["time"])
        wysokosci_m, temperatura, wiatr_u, wiatr_v, cisnienie, zmienne_dynamiczne = (
            przygotuj_dane_wysokosciowe(dane_api, elewacja_m, poziomy_hpa)
        )

        # Po udanym pobraniu zapamiętujemy dane i usuwamy poprzednią zawartość okna.
        ostatnie_zmienne = zmienne_dynamiczne
        for widget in getattr(fig, "_weather_widgets", []):
            widget.disconnect_events()
        poprzednie_klikniecie = getattr(fig, "_weather_click_id", None)
        if poprzednie_klikniecie is not None:
            fig.canvas.mpl_disconnect(poprzednie_klikniecie)
        fig.clear()

        # Matplotlib zapisuje daty jako liczby, co ułatwia odnalezienie godziny kliknięcia.
        czasy_numeryczne = matplotlib.dates.date2num(czasy.to_pydatetime())
        X, Y = np.meshgrid(czasy_numeryczne, wysokosci_m)

        # Lewą, większą część okna przeznaczamy na wykres.
        ax = fig.add_axes([0.07, 0.17, 0.67, 0.74])
        contour = ax.contourf(X, Y, temperatura, levels=20, cmap="RdYlBu_r", alpha=0.8)
        fig.colorbar(contour, ax=ax, label="Temperatura (°C)")

        # Czarne linie łączą punkty o takim samym ciśnieniu.
        izobary = ax.contour(
            X, Y, cisnienie, levels=np.arange(725, 1001, 25),
            colors="black", linewidths=0.7, alpha=0.75
        )
        ax.clabel(izobary, inline=True, fontsize=8, fmt="%1.0f hPa")

        # Strzałki pokazują wiatr; pomijamy co drugą próbkę, aby nie zagęścić rysunku.
        wiatr = ax.quiver(
            X[::2, ::2], Y[::2, ::2], wiatr_u[::2, ::2], wiatr_v[::2, ::2],
            color="black", scale=250, headwidth=4, alpha=0.8
        )
        ax.quiverkey(wiatr, 0.88, 1.03, 10, "10 m/s", labelpos="E", coordinates="axes")

        # Opisujemy osie i pokazujemy nazwę wybranego modelu oraz współrzędne.
        nazwa_modelu = next(etykieta for etykieta, kod in modele.items() if kod == model)
        ax.set_title(
            f"Prognoza: {nazwa_modelu} | {szerokosc_geograficzna:.3f}, "
            f"{dlugosc_geograficzna:.3f}\nWysokość 0-2500 m nad gruntem",
            fontsize=12,
        )
        ax.set_ylabel("Wysokość nad gruntem (metry)")
        ax.set_xlabel("Czas")
        ax.set_ylim(0, 2500)
        ax.xaxis.set_major_formatter(DateFormatter("%d.%m %H:00"))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.3)

        # Przygotowujemy dymek; pojawi się dopiero po kliknięciu na wykres.
        szczegoly = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "black", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "black"},
            fontsize=9,
            zorder=10,
        )
        szczegoly.set_visible(False)

        # Skróty 16 kierunków kompasu, od północy zgodnie z ruchem wskazówek zegara.
        kierunki_kompasu = (
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
        )

        def pokaz_szczegoly(event):
            """Pokazuje dane dla najbliższego punktu prognozy po kliknięciu."""
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            indeks_czasu = int(np.argmin(np.abs(czasy_numeryczne - event.xdata)))
            indeks_wysokosci = int(np.argmin(np.abs(wysokosci_m - event.ydata)))
            wysokosc = wysokosci_m[indeks_wysokosci]
            predkosc = np.hypot(
                wiatr_u[indeks_wysokosci, indeks_czasu],
                wiatr_v[indeks_wysokosci, indeks_czasu],
            )
            kierunek = np.mod(np.rad2deg(np.arctan2(
                -wiatr_u[indeks_wysokosci, indeks_czasu],
                -wiatr_v[indeks_wysokosci, indeks_czasu],
            )), 360)
            kierunek_kompasowy = kierunki_kompasu[int(round(kierunek / 22.5)) % 16]

            szczegoly.xy = (czasy_numeryczne[indeks_czasu], wysokosc)
            szczegoly.set_text(
                f"Wysokość: {wysokosc} m n.p.t.\n"
                f"Data: {czasy[indeks_czasu]:%d.%m.%Y %H:%M}\n"
                f"Temperatura: {temperatura[indeks_wysokosci, indeks_czasu]:.1f} °C\n"
                f"Wiatr: {predkosc:.1f} m/s\n"
                f"Kierunek (skąd wieje): {kierunek:.0f}° ({kierunek_kompasowy})\n"
                f"Ciśnienie: {cisnienie[indeks_wysokosci, indeks_czasu]:.1f} hPa"
            )
            szczegoly.set_visible(True)
            fig.canvas.draw_idle()

        # Podpinamy obsługę kliknięcia do aktualnego wykresu.
        fig._weather_click_id = fig.canvas.mpl_connect("button_press_event", pokaz_szczegoly)

        # Prawy panel zawiera wybór modelu, pola współrzędnych i przycisk odświeżenia.
        fig.text(0.78, 0.93, "USTAWIENIA", fontsize=12, weight="bold")
        fig.text(0.78, 0.88, "Model prognozy:", fontsize=10)
        etykiety_modeli = list(modele)
        aktywny_model = etykiety_modeli.index(nazwa_modelu)
        radio = RadioButtons(
            fig.add_axes([0.78, 0.72, 0.20, 0.14]),
            etykiety_modeli,
            active=aktywny_model,
        )
        # Opis wyjaśnia, dlaczego automat jest polecany dla lokalizacji startu.
        fig.text(0.78, 0.63, "ZALECENIE DLA MISJI", fontsize=9, weight="bold")
        fig.text(
            0.78, 0.56,
            "Automatyczny dobór Open-Meteo wybiera model o najwyższej dostępnej "
            "rozdzielczości. GFS służy tu jako porównanie.",
            fontsize=8,
            wrap=True,
        )

        # Wpisujemy szerokość geograficzną (latitude), od -90 do 90 stopni.
        pole_szerokosci = TextBox(
            fig.add_axes([0.78, 0.45, 0.19, 0.06]),
            "Szerokość (lat): ",
            initial=f"{szerokosc_geograficzna:.4f}",
        )
        # Wpisujemy długość geograficzną (longitude), od -180 do 180 stopni.
        pole_dlugosci = TextBox(
            fig.add_axes([0.78, 0.35, 0.19, 0.06]),
            "Długość (lon): ",
            initial=f"{dlugosc_geograficzna:.4f}",
        )
        # Kliknięcie przycisku pobiera prognozę z aktualnie wybranymi ustawieniami.
        przycisk = Button(fig.add_axes([0.78, 0.24, 0.19, 0.07]), "Przeładuj prognozę")
        status = fig.text(0.78, 0.13, "", fontsize=9, color="darkred", wrap=True)

        def przeladuj_prognoze(event):
            """Sprawdza wpisane ustawienia i pobiera nową prognozę."""
            try:
                # Zamieniamy wpisane liczby na wartości i akceptujemy przecinek dziesiętny.
                nowa_szerokosc = float(pole_szerokosci.text.replace(",", "."))
                nowa_dlugosc = float(pole_dlugosci.text.replace(",", "."))
                if not -90 <= nowa_szerokosc <= 90:
                    raise ValueError("Szerokość musi być od -90 do 90.")
                if not -180 <= nowa_dlugosc <= 180:
                    raise ValueError("Długość musi być od -180 do 180.")

                wybrany_model = modele[radio.value_selected]
                status.set_text("Pobieranie prognozy...")
                fig.canvas.draw_idle()
                rysuj_prognoze(nowa_szerokosc, nowa_dlugosc, wybrany_model)
            except (ValueError, requests.RequestException, KeyError) as blad:
                # Błędny wpis lub problem z internetem pokazujemy w panelu, bez zamykania okna.
                status.set_text(str(blad))
                fig.canvas.draw_idle()

        przycisk.on_clicked(przeladuj_prognoze)

        # Przechowujemy widgety, aby Python nie usunął ich przed kliknięciem.
        fig._weather_widgets = [radio, pole_szerokosci, pole_dlugosci, przycisk]
        fig.canvas.draw_idle()

    # Pierwszy wykres używa domyślnych współrzędnych i automatycznego modelu.
    rysuj_prognoze(szerokosc_startowa, dlugosc_startowa, None)
    plt.show()

    # Zwracamy dane z ostatnio wyświetlonej prognozy po zamknięciu okna.
    return ostatnie_zmienne


# Ten warunek uruchamia aplikację tylko wtedy, gdy plik jest uruchomiony bezpośrednio.
# Dzięki temu można też zaimportować funkcje z tego pliku do innego programu bez
# otwierania wykresu od razu.
if __name__ == "__main__":
    glowna_aplikacja()