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

Seit der Erweiterung fließen drei weitere Maße ein: **EV/EBITDA** (unabhängig
von Kapitalstruktur und Abschreibungen), die **Cashflow-Rendite** (schwerer zu
schönen als der Gewinn) und das **KBV** (greift auch dort, wo der Gewinn gerade
nichts aussagt). Entscheidend ist die **Bewertungsbreite**: Ein Titel, der nur
beim KGV billig aussieht, hat meist Einmaleffekte im Gewinn — erst wenn alle
vier Maße unter dem Branchenschnitt liegen, ist die Bewertung wirklich niedrig.
Der Report weist das als „3/4 Maße günstig" aus.

Steigt das erwartete über das aktuelle KGV, markiert der Report das explizit als
Value-Falle. Der Branchenmedian wird über das gesamte Universum gebildet, nicht
nur über die Treffer — sonst vergliche man günstige Titel nur mit anderen
günstigen Titeln.

### 2. Chart-Analyse

Beschränkt auf das, was sich sauber berechnen lässt: Trendstruktur (SMA 50/200),
Momentum (RSI, MACD, Stochastik), Volatilität (realisiert, ATR, Bollinger-Bänder),
Volumenverhalten und relative Stärke gegen den jeweiligen Index. **Keine** Mustererkennung wie Schulter-Kopf-Schulter —
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

Dazu kommen fünf Kennzahlen, die direkt aus Bilanz und GuV gelesen werden und
in den Übersichtsdaten von Yahoo nicht enthalten sind:

| Kennzahl | Warum sie zählt |
|---|---|
| **ROIC** | Rendite auf das eingesetzte Kapital. Anders als die Eigenkapitalrendite lässt sie sich nicht durch Verschuldung schönen: Wer Eigenkapital durch Kredite ersetzt, hebt den ROE, den ROIC aber nicht. |
| **Zinsdeckung** | EBIT geteilt durch die Zinslast. Unter 2 trifft ein Gewinnrückgang direkt die Fähigkeit, Kredite zu bedienen. |
| **Liquiditätsgrad** | Umlaufvermögen zu kurzfristigen Verbindlichkeiten. Unter 1 fehlt Deckung. |
| **Margentrend** | Die Richtung der Nettomarge über die letzten Quartale, nicht ihr Stand — ein Frühindikator, den der ausgewiesene Gewinn erst verzögert zeigt. |
| **Verwässerung** | Jährliche Veränderung der Aktienzahl. Wächst sie, wächst der Gewinn je Aktie langsamer als der Gewinn; schrumpft sie, laufen Rückkäufe. |

Wo ein Titel diese Daten nicht hergibt — bei kleinen Nebenwerten ist das die
Regel — fehlen die Kennzahlen einfach. Der Score bestraft das nicht.

### 4. Makrokontext

Aus 16 FRED-Zeitreihen — Leitzins, Renditekurve, Inflation und
Inflationserwartung, Arbeitsmarkt, Industrieproduktion, BIP,
Verbrauchervertrauen, Öl, VIX und dem Risikoaufschlag für Hochzinsanleihen —
werden Zinsrichtung, Kurvenform, Inflations- und Wachstumssignal abgeleitet und
über eine fest kodierte Sensitivitätstabelle in Sektor-Scores übersetzt. Der
Teil ist deterministisch und in `macro/regime.py` Zeile für Zeile nachrechenbar.

Weil mehr als die Hälfte des Universums in Europa notiert, wäre ein rein
amerikanisches Makrobild der falsche Maßstab. Deshalb kommen zwei europäische
Quellen dazu — beide **ohne API-Key**:

* **Eurostat**: Industrieproduktion, Arbeitslosenquote, BIP und HVPI-Inflation
  für den Euroraum.
* **EZB Data Portal**: Hauptrefinanzierungssatz sowie die Renditen zehn- und
  zweijähriger Euro-Staatsanleihen; daraus wird die europäische Renditekurve
  abgeleitet.

Zinsrichtung und Kurvenform mitteln beide Wirtschaftsräume, das Wachstumssignal
zieht europäische und amerikanische Konjunkturdaten gleichberechtigt heran. Für
die Bewertung gegen die Anleiherendite bekommt ein europäischer Titel die
Euroraum-Rendite, ein amerikanischer die US-Rendite. Fällt eine Quelle aus,
fehlt sie im Bild und der Lauf geht weiter.

### 5. LLM-Einordnung

Für jeden Treffer beantwortet Claude die Frage, die sich aus Zahlen allein nicht
beantworten lässt: *Warum* ist dieser Titel billig, und passt der Grund zum
Umfeld? Das Modell bekommt die fertig gerechneten Kennzahlen, das Makrobild und
aktuelle Schlagzeilen — es rechnet nichts nach und gibt keine Empfehlung ab.

Ergebnis ist eine strukturierte Einordnung mit einem von drei Urteilen:
`zyklisch-guenstig`, `strukturell-billig` oder `unklar`, plus Konfidenz.

### 6. Hebelprodukte — Risikorechnung auf den Basiswert

Das Werkzeug **screent keine Hebelprodukte**, und das ist eine bewusste
Entscheidung. Ein Knock-out hat kein KGV, keine Bilanz und keinen Sektor; sein
Preis ist eine mechanische Funktion des Basiswerts, der Finanzierungskosten und
des Emittenten-Spreads. Das Fundamentalmodul ist auf ihn nicht anwendbar, und
die Produktdaten der deutschen Emittenten liegen in keiner der genutzten
Quellen — pro Basiswert existieren oft tausende WKNs.

Was es stattdessen tut: Für jeden Treffer ausrechnen, **welcher Hebel zur
Schwankungsbreite dieses Titels überhaupt passt**.

```bash
broker leverage SAP.DE --factor 4 --days 60
```

| Rechnung | Warum sie zählt |
|---|---|
| **Volatilitätsdrag** | Ein Faktor-4-Papier verliert *mechanisch* Geld, wenn der Basiswert schwankt — auch bei unverändertem Endstand |
| **Knock-out-Wahrscheinlichkeit** | Nicht ob der Kurs am Ende unter der Schwelle liegt, sondern ob er sie *unterwegs berührt*. Das ist doppelt so wahrscheinlich |
| **Maximal sinnvoller Hebel** | aus Volatilität, Haltedauer und Verlusttoleranz |
| **Finanzierungskosten** | (k−1) × Zins über die Haltedauer, mit dem Zinsniveau aus dem Makromodul |

Zur Größenordnung, bei 60 Handelstagen Haltedauer und Faktor 4:

| Volatilität des Basiswerts | Kosten bei unverändertem Kurs | Knock-out bei 20 % Abstand |
|---|---|---|
| 22 % (ruhiger Standardwert) | −11 % | 4 % |
| 35 % (typische Aktie) | −20 % | 19 % |
| 60 % (volatiler Nebenwert) | −45 % | 45 % |

Alle Zahlen unterstellen konstante Volatilität und keine Kurssprünge. Echte
Kurse springen — die tatsächliche Knock-out-Gefahr liegt eher *über* diesen
Werten. Es sind Untergrenzen, keine Prognosen.

> Ein Hinweis zur Zeitachse: Das Bewertungssignal dieses Screeners zielt auf
> Monate. Ein Knock-out zahlt täglich Finanzierungskosten und kann durch eine
> Zwischenbewegung ausgestoppt werden, auch wenn die These am Ende aufgeht.
> Ein langsames Signal mit einem schnellen Instrument umzusetzen ist keine
> Verstärkung, sondern eine andere Wette.

### 7. Plausibilitätsprüfung der Rohdaten

Die riskanteste Stelle des ganzen Werkzeugs: Ein falsches KGV erzeugt keinen
Fehler, sondern einen besonders attraktiven Treffer — genau den, der oben auf
der Liste landet und den man kauft.

Solche Fehler lassen sich billig erkennen, weil die Kennzahlen redundant sind.
Kurs geteilt durch Gewinn je Aktie *muss* das gemeldete KGV ergeben, Kurs mal
Aktienzahl *muss* die Marktkapitalisierung ergeben. Wo das nicht aufgeht,
stimmt etwas nicht — und man weiß es, ohne die Wahrheit zu kennen.

| Prüfung | Findet |
|---|---|
| KGV gegen Kurs/Gewinn | nicht verarbeiteter Aktiensplit, veralteter Gewinn |
| Erwartetes KGV gegen Kurs/Schätzung | dasselbe für Analystendaten |
| Marktkapitalisierung gegen Kurs × Aktienzahl | Kennzahlen in verschiedenen Währungen, veraltete Aktienzahl |
| Alter des letzten Kurses | ausgesetzter Handel, Delisting |
| Größter Tagessprung | Split-Artefakt oder echter Kurssturz — beides verzerrt die Chart-Kennzahlen |

Betroffene Titel **fliegen nicht raus**. Sie werden im Report benannt und im
Score gedämpft (auf 90 % bei Auffälligkeiten, auf 60 % bei echten
Widersprüchen). Ein stiller Ausschluss würde genau die Fälle verdecken, die man
sehen will — und ein Titel soll nicht wegen eines falschen KGV nach oben
rutschen.

**Ein erklärter Sonderfall kostet nichts.** Bei deutschen Vorzugsaktien —
VOW3, HEN3, DRW3, JUN3, KSB3 — meldet Yahoo die Marktkapitalisierung des
ganzen Unternehmens, die Aktienzahl aber nur für die Vorzugsgattung. Die
Prüfung sah darin einen Widerspruch von 50 bis 63 % und strich 40 % vom Score,
obwohl kein Fehler vorlag: Der Faktor entspricht exakt dem Verhältnis aller
Aktien zu denen dieser Gattung. Deshalb entscheidet jetzt die **Richtung** der
Abweichung:

* Gemeldete Marktkapitalisierung **größer** als Kurs × Aktienzahl → mehrere
  Aktiengattungen. Erscheint als Hinweis im Report, kostet keine Punkte, denn
  EV/EBITDA und FCF-Rendite setzen Unternehmenswert und Unternehmenszahlen ins
  Verhältnis — beide Seiten meinen das ganze Unternehmen. Ab dem Fünffachen
  greift die Erklärung nicht mehr und der Befund wird wieder hart.
* Gemeldete Marktkapitalisierung **kleiner** → Währungsproblem oder veraltete
  Zahl. Das verzerrt die Kennzahlen tatsächlich und bleibt ein harter Befund.

Dieselbe Verwechslung steckte in der historischen KGV-Reihe: Wo Yahoo kein
EPS ausweist, wird es aus Nettogewinn und Aktienzahl gerechnet — der Gewinn des
ganzen Unternehmens geteilt durch die Aktien einer Gattung ergibt bei VW das
2,4-fache des echten EPS. Das historische KGV fiel entsprechend zu niedrig aus,
das aktuelle KGV lag scheinbar über dem gesamten Verlauf, und der Titel bekam
im Perzentil-Vergleich null Punkte. Jetzt wird die zum Gesamtwert passende
Aktienzahl verwendet.

### 8. Journal — die Vorwärtsmessung

Jeder Lauf hält seine Treffer mit Kurs, Datum und allen Teilscores in
`journal/history.jsonl` fest. Spätere Läufe rechnen aus, wie sich diese Titel
seither gegenüber ihrem Index entwickelt haben.

Das ist **kein Backtest**, und das ist Absicht. Ein Backtest bräuchte
Fundamentaldaten zum damaligen Stand; was heute in einer Gratis-Datenbank steht,
ist die nachträglich korrigierte Fassung. Wer damit rückrechnet, weiß Dinge, die
zum Kaufzeitpunkt niemand wusste, und bekommt systematisch zu schöne Ergebnisse.
Hier wird stattdessen nach vorne gemessen — langsamer, aber verzerrungsfrei.

```bash
broker track           # Auswertung über 1, 3, 6 und 12 Monate
broker track --list    # nur die Beständigkeit je Titel
```

Zwei Vorkehrungen gegen Selbstbetrug stecken fest im Code:

- **Entprellung.** Derselbe Titel steht oft wochenlang in Folge auf der Liste.
  Zählte man jede Nennung, entstünden aus einer einzigen Kursbewegung dreißig
  scheinbar unabhängige Datenpunkte. Für die Auswertung zählt pro Titel und
  Kalendermonat nur die erste Nennung.
- **Feste Auswertungsraster.** Die Gruppen (Score-Bänder, LLM-Urteil,
  Chart-Setup) stehen im Code fest. Wer so lange nach Schnittmengen sucht, bis
  eine gut aussieht, findet immer eine. Gruppen unter zehn Beobachtungen werden
  gar nicht erst ausgewiesen.

Ausgewiesen wird der **Median** der Überrendite, nicht der Mittelwert:
Aktienrenditen sind stark rechtsschief, ein einzelner Verdoppler würde den
Mittelwert dominieren.

Aus dem Journal fällt nebenbei die **Beständigkeit** ab — in wie vielen der
letzten 20 Läufe ein Titel auf der Liste stand. Ein Titel, der einmalig aufblitzt
und tags darauf verschwindet, geht häufiger auf einen Datenfehler zurück als auf
eine echte Fehlbewertung.

> Realistische Erwartung: Vor etwa einem Jahr Laufzeit ist hier nichts
> statistisch belastbar. Die Zahlen davor sind eine kleine Stichprobe, kein
> Beleg.

## Konfiguration

Alles über Umgebungsvariablen oder `.env` (siehe `.env.example`).

| Variable | Nötig für | Kosten |
|---|---|---|
| — | KGV-, Chart-, Qualitätsanalyse, europäischer Makrokontext (Eurostat, EZB) | kostenlos |
| `FRED_API_KEY` | US-Makrodaten im Sektor-Score | kostenlos |
| `ANTHROPIC_API_KEY` | LLM-Einordnung | wenige Euro/Monat |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | Benachrichtigung aufs Handy | kostenlos |
| `SMTP_*` | Benachrichtigung per Mail | — |

Ohne `FRED_API_KEY` stützt sich das Makrobild allein auf Eurostat und EZB — es
bleibt also nutzbar, verliert aber die amerikanische Seite. Ist gar keine Quelle
erreichbar, wird der Makro-Teil neutral bewertet und im Report als solcher
gekennzeichnet — er täuscht keine Daten vor, die er nicht hat.

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
broker leverage SAP.DE --factor 4 --days 60     # welcher Hebel passt?
broker track                                   # Journal auswerten
broker track --list                            # Beständigkeit je Titel
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

Für den Makrokontext kommen drei weitere Quellen dazu:

| Quelle | Was sie liefert | Key |
|---|---|---|
| FRED (St. Louis Fed) | 16 US-Reihen: Leitzins, Renditekurve, Inflation, Arbeitsmarkt, Industrieproduktion, BIP, Öl, VIX, Risikoaufschläge | kostenlos, Registrierung |
| Eurostat | Euroraum: Industrieproduktion, Arbeitslosenquote, BIP, HVPI | keiner |
| EZB Data Portal | Euroraum: Leitzins, Renditen 10J und 2J | keiner |

Alle drei sind optional. Fällt eine einzelne Reihe aus, erscheint sie als
Warnung im Log und die übrigen laufen weiter.

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

309 Tests, alle mit synthetischen Daten und ohne Netzwerkzugriff.

## Aufbau

```
src/broker/
  config.py          Konfiguration, Gewichte, Schwellenwerte
  models.py          Datenmodelle zwischen den Schichten
  screener.py        Ablauf: Daten holen, filtern, bewerten, ranken
  cli.py             Kommandozeile
  universe/          Indexlisten und Loader
  providers/         Marktdaten (Schnittstelle + yfinance + Cache)
  analysis/          Bewertung, Chart, Qualität, Datenplausibilität,
                     Hebelrisiko, Gesamtscore
  macro/             FRED, Eurostat und EZB, Regime-Ableitung,
                     Sektor-Sensitivität
  context/           News-Beschaffung, LLM-Einordnung
  journal.py         Vorwärtsmessung: was wurde wann vorgeschlagen, wie lief es
  report/            HTML-Report und JSON-Export
journal/history.jsonl  Aufzeichnung (wird eingecheckt, wächst mit jedem Lauf)
```
