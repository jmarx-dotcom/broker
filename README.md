# broker

Ein Screener, der auf günstig bewertete Aktien aufmerksam macht — über eine
KGV-Analyse, eine Chart-Analyse und den aktuellen volkswirtschaftlichen Kontext.

Universum: DAX, MDAX, SDAX, Euro Stoxx und S&P 500 (rund 680 Titel).

> **Kein Anlageprodukt.** Das Tool liest ausschließlich öffentliche Marktdaten
> und erzeugt einen Report. Es hat keine Verbindung zu einem Depot und kann
> keine Orders auslösen. Die Treffer sind Rechercheergebnisse, keine Empfehlungen.

## Schnellstart

```bash
pip install -e .
cp .env.example .env          # Keys eintragen (optional, siehe unten)

broker universe --list        # verfügbare Gruppen anzeigen
broker macro                  # aktuelles Makrobild
broker screen --universe dax --report
```

Ohne jeden API-Key läuft der komplette Screener inklusive KGV-, Chart- und
Qualitätsanalyse. Die Keys schalten den Makro-Teil und die LLM-Einordnung frei.

## Was das Tool macht

### 1. KGV-Analyse

Ein niedriges KGV ist für sich genommen kein Kaufsignal — meistens ist es eine
Warnung: Der Markt preist einbrechende Gewinne ein, bevor sie in den Zahlen
stehen. "Günstig" ist deshalb immer relativ definiert:

| Vergleich | Was er beantwortet |
|---|---|
| KGV-Perzentil der eigenen 3-Jahres-Historie | Ist der Titel *für sich selbst* billig? |
| KGV gegen den Branchenmedian | Ist er billig *im Vergleich zu seinesgleichen*? |
| Forward- gegen Trailing-KGV | Erwarten Analysten steigende oder fallende Gewinne? |
| Gewinnrendite gegen Anleiherendite | Gibt es überhaupt eine Prämie fürs Aktienrisiko? |

Steigt das erwartete über das aktuelle KGV, markiert der Report das explizit als
Value-Falle. Der Branchenmedian wird über das gesamte Universum gebildet, nicht
nur über die Treffer — sonst vergliche man günstige Titel nur mit anderen
günstigen Titeln.

### 2. Chart-Analyse

Beschränkt auf das, was sich sauber berechnen lässt: Trendstruktur (SMA 50/200),
Momentum (RSI, MACD), Volatilität, Volumenverhalten und relative Stärke gegen
den jeweiligen Index. **Keine** Mustererkennung wie Schulter-Kopf-Schulter —
dafür gibt es keine belastbare Evidenz.

Der Chart-Teil hat genau eine Aufgabe: unterscheiden, ob ein günstiger Titel
gerade ausverkauft ist oder noch fällt.

| Setup | Bedeutung |
|---|---|
| Bodenbildung nach Rücksetzer | tief unterm Hoch, Momentum dreht — der interessante Fall |
| überverkauft, Trendwende offen | tief unterm Hoch, RSI niedrig, Abwärtsdruck lässt nach |
| fallendes Messer | tief unterm Hoch, fällt weiter — hier ist das niedrige KGV eine Falle |
| intakter Aufwärtstrend | über dem 200er-Schnitt mit Momentum |

### 3. Qualitätsfilter

Ohne diesen Schritt findet ein KGV-Screener zuverlässig die Titel, die zu Recht
billig sind. Geprüft werden Nettoverschuldung/EBITDA, Umsatz- und
Gewinnentwicklung, Eigenkapitalrendite, Marge, freier Cashflow und
Ausschüttungsquote. Jedes Problem erscheint als benanntes Warnsignal im Report,
nicht nur als Punktabzug.

### 4. Makrokontext

Aus FRED-Zeitreihen (Leitzins, Renditekurve, Inflation, Arbeitsmarkt, Öl, VIX)
werden Zinsrichtung, Kurvenform, Inflations- und Wachstumssignal abgeleitet und
über eine fest kodierte Sensitivitätstabelle in Sektor-Scores übersetzt. Der
Teil ist deterministisch und in `macro/regime.py` Zeile für Zeile nachrechenbar.

### 5. LLM-Einordnung

Für jeden Treffer beantwortet Claude die Frage, die sich aus Zahlen allein nicht
beantworten lässt: *Warum* ist dieser Titel billig, und passt der Grund zum
Umfeld? Das Modell bekommt die fertig gerechneten Kennzahlen, das Makrobild und
aktuelle Schlagzeilen — es rechnet nichts nach und gibt keine Empfehlung ab.

Ergebnis ist eine strukturierte Einordnung mit einem von drei Urteilen:
`zyklisch-guenstig`, `strukturell-billig` oder `unklar`, plus Konfidenz.

## Konfiguration

Alles über Umgebungsvariablen oder `.env` (siehe `.env.example`).

| Variable | Nötig für | Kosten |
|---|---|---|
| — | KGV-, Chart-, Qualitätsanalyse | kostenlos |
| `FRED_API_KEY` | Makrokontext und Sektor-Scores | kostenlos |
| `ANTHROPIC_API_KEY` | LLM-Einordnung | wenige Euro/Monat |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | Benachrichtigung aufs Handy | kostenlos |
| `SMTP_*` | Benachrichtigung per Mail | — |

Ohne `FRED_API_KEY` wird der Makro-Teil neutral bewertet und im Report als
solcher gekennzeichnet — er täuscht keine Daten vor, die er nicht hat.

Umschließender Leerraum wird bei allen Keys automatisch entfernt. Beim
Kopieren in Web-Formulare rutscht regelmäßig ein Leerzeichen mit ans Ende, und
die Gegenstelle antwortet dann mit einem nichtssagenden `400` oder `401` auf
einen Key, der im maskierten Log völlig korrekt aussieht.

Ob die Benachrichtigung funktioniert, prüfst du ohne kompletten Lauf:

```bash
broker notify          # zeigt eingerichtete Kanäle, validiert den Bot-Token
broker notify --send   # verschickt zusätzlich eine Testnachricht
```

## Befehle

```bash
broker screen --universe europe --report --json --notify
broker screen --universe all --no-llm          # nur Kennzahlen, kein LLM
broker screen --universe dax --ticker ROCK.CO  # Index plus eigene Titel
broker screen --universe DAX,SP500 --limit 50  # eigene Indexkombination
broker macro                                   # Makrobild und Sektor-Scores
broker universe germany                        # Titel einer Gruppe auflisten
broker notify                                  # Benachrichtigung prüfen
broker notify --send                           # Testnachricht verschicken
broker cache --keep-days 3                     # alte Cache-Tage löschen
```

Gruppen: `dax`, `mdax`, `sdax`, `germany`, `estoxx`, `europe`, `sp500`, `us`,
`all` — oder eine kommagetrennte Indexliste.

## Datenquellen

Standardanbieter ist **yfinance**: kostenlos, aber eine inoffizielle
Schnittstelle mit Lücken (besonders bei Forward-EPS europäischer Nebenwerte) und
gelegentlichen Ausfällen. Für den produktiven Dauerbetrieb ist ein
Bezahlanbieter die bessere Wahl.

Der Wechsel ist vorbereitet: `providers/base.py` definiert die Schnittstelle mit
drei Methoden (`fundamentals`, `history`, `news`). Eine FMP- oder
EODHD-Implementierung ist eine neue Datei in `providers/` und eine Zeile in
`providers/factory.py` — der Rest des Codes bleibt unberührt.

Ein Tagescache unter `cache/` verhindert, dass ein wiederholter Lauf dieselben
tausend Requests noch einmal stellt.

## Automatischer Betrieb

`.github/workflows/screening.yml` startet werktags einen Lauf und legt den
Report als Artifact ab. Die Keys gehören als GitHub Secrets hinterlegt, nicht in
den Code. Für Benachrichtigungen zusätzlich `TELEGRAM_BOT_TOKEN` und
`TELEGRAM_CHAT_ID` als Secrets setzen.

## Grenzen

- **Die Indexlisten sind Snapshots.** Index-Zugehörigkeiten ändern sich mehrmals
  im Jahr; ein Ticker, den es nicht mehr gibt, erzeugt eine Warnung im Log und
  wird übersprungen.
- **Die historische KGV-Reihe ist eine Näherung.** Sie wird aus Quartalsgewinnen
  und der heutigen Aktienzahl gebildet; Rückkäufe verzerren sie leicht. Für
  einen Perzentilvergleich reicht das, für eine Bewertung auf zwei
  Nachkommastellen nicht.
- **Die Sektor-Sensitivität ist eine grobe Heuristik** über breit belegte
  Zusammenhänge, keine Einzelfallanalyse. Sie geht deshalb nur mit 10 % in den
  Gesamtscore ein.
- **Kein Backtest.** Die Gewichte in `config.py` sind begründet gesetzt, nicht
  optimiert. Wer sie optimieren will, braucht Punkt-in-der-Zeit-Fundamentaldaten
  — yfinance liefert die nicht.

## Tests

```bash
pytest
```

84 Tests, alle mit synthetischen Daten und ohne Netzwerkzugriff.

## Aufbau

```
src/broker/
  config.py          Konfiguration, Gewichte, Schwellenwerte
  models.py          Datenmodelle zwischen den Schichten
  screener.py        Ablauf: Daten holen, filtern, bewerten, ranken
  cli.py             Kommandozeile
  universe/          Indexlisten und Loader
  providers/         Marktdaten (Schnittstelle + yfinance + Cache)
  analysis/          KGV, Chart, Qualität, Gesamtscore
  macro/             FRED, Regime-Ableitung, Sektor-Sensitivität
  context/           News-Beschaffung, LLM-Einordnung
  report/            HTML-Report und JSON-Export
```
