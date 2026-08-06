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

**Das Werkzeug verändert sich dabei nicht selbst.** Die Gewichte in `config.py`
bleiben, wo sie sind; `combine_scores()` bekommt das Journal gar nicht zu sehen.
Das ist Absicht — eine Automatik, die Gewichte an wenige Dutzend Beobachtungen
anpasst, optimiert auf Rauschen. Das Journal sammelt Belege; was man daraus
schließt, bleibt eine bewusste Entscheidung.

Das ist **kein Backtest**, und das ist Absicht. Ein Backtest bräuchte
Fundamentaldaten zum damaligen Stand; was heute in einer Gratis-Datenbank steht,
ist die nachträglich korrigierte Fassung. Wer damit rückrechnet, weiß Dinge, die
zum Kaufzeitpunkt niemand wusste, und bekommt systematisch zu schöne Ergebnisse.
Hier wird stattdessen nach vorne gemessen — langsamer, aber verzerrungsfrei.

```bash
broker track           # Auswertung über 1, 3, 6 und 12 Monate
broker track --list    # nur die Beständigkeit je Titel
```

**Die Kontrollgruppe.** Jeder Lauf schreibt zusätzlich 15 zufällig gezogene
Titel mit, die es *nicht* auf die Liste geschafft haben. Ohne sie könnte das
Journal nur zeigen, wie die Vorschläge gelaufen sind — nicht, ob die Auswahl
überhaupt etwas taugt. Erst der Vergleich beantwortet die eigentliche Frage:

```
Treffer gegen Kontrollgruppe
  Treffer           n=34   Median-Überrendite  +3.1%   Trefferquote 0.59
  Kontrollgruppe    n=31   Median-Überrendite  +0.4%   Trefferquote 0.48
    Vorsprung der Auswahl: +2.7 Prozentpunkte
```

Die Ziehung ist an das Datum gebunden: Ein wiederholter Lauf am selben Tag zieht
dieselbe Gruppe. Sonst ließe sich — auch ungewollt — so lange neu würfeln, bis
die Kontrollgruppe schlecht aussieht, und genau die Beliebigkeit wäre eingebaut,
gegen die sie schützen soll. Der Vorsprung wird erst ausgewiesen, wenn **beide**
Gruppen zehn Beobachtungen haben; ein Vorsprung gegenüber drei Kontrolltiteln
ist keiner.

Zwei weitere Vorkehrungen gegen Selbstbetrug stecken fest im Code:

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
broker notify          # Kanäle, Bot-Token und Chat-ID prüfen
broker notify --send   # verschickt zusätzlich eine Testnachricht
```

Telegram braucht zwei Angaben, und ein gültiger Token sagt nichts über die
zweite. Deshalb prüft `broker notify` beide: Über `getUpdates` liest es die
Chats, aus denen der Bot zuletzt gehört hat, und vergleicht sie mit
`TELEGRAM_CHAT_ID`. Passt die Nummer nicht, nennt die Ausgabe die Nummern, die
passen würden — dasselbe steht im Log, wenn ein Versand mit
`chat not found` scheitert.

Bleibt die Liste leer, hat der Bot noch nie eine Nachricht bekommen. Ein Bot
darf niemanden von sich aus anschreiben: Chat öffnen, einmal **Start** drücken.
Bei einem **neu angelegten Bot zählt der alte Chat nicht** — Start muss für den
neuen Bot noch einmal gedrückt werden, auch wenn die eigene Chat-ID dieselbe
bleibt.

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
broker doctor                                  # Quellen auf Drift prüfen
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

`.github/workflows/screening.yml` startet werktags um 18:30 UTC einen Lauf über
das **gesamte** Universum (`all`, rund 680 Titel) und legt den Report als
Artifact ab. Die Keys gehören als GitHub Secrets hinterlegt, nicht in den Code.
Für Benachrichtigungen zusätzlich `TELEGRAM_BOT_TOKEN` und `TELEGRAM_CHAT_ID`
als Secrets setzen.

Die Universumsgröße kostet Laufzeit, aber kaum Geld: `max_candidates` deckelt
die Trefferliste bei 15, und nur diese 15 gehen ans Modell. Der Sprung von
`europe` auf `all` verlängert einen Lauf von rund viereinhalb auf etwa acht
Minuten, die LLM-Kosten bleiben gleich.

Der Zeitplan übergibt keine Eingaben — über das Universum des täglichen Laufs
entscheidet deshalb der Ersatzwert in der `run`-Zeile, nicht der `default` unter
`workflow_dispatch`. Beide stehen auf `all`; wer das ändert, muss beide ändern.

**Geplante Läufe kommen zu spät, und zwar systematisch.** GitHub stellt
`schedule`-Läufe in eine Warteschlange und liefert sie aus, wenn Kapazität frei
ist. Bisher lag der Verzug bei 90 bis 99 Minuten. Dagegen lässt sich nichts
einstellen — wer die Nachricht zu einer bestimmten Zeit braucht, muss den Cron
entsprechend früher setzen.

### Ein gestörter Lauf sagt das

Am 4. und 5. August meldete das Werkzeug zwei Tage hintereinander „keine Treffer
über der Score-Schwelle". Tatsächlich waren von 674 Titeln nur 342 mit Daten
angekommen, und von 217 Titeln, die die harten Filter passiert hatten, wurde
**genau einer** bewertet — 548 Abrufe hatte Yahoo abgewiesen. Der DAX-Lauf über
40 Titel am selben Code hatte null Fehler. Nicht die Titel waren das Problem,
sondern ihre Zahl.

Die Nachricht war beruhigend und falsch, und ohne einen Blick ins Log war ihr
das nicht anzusehen. Das ist dieselbe Fehlerart wie „Kein Benachrichtigungskanal
konfiguriert" und „Pull Request existiert bereits": eine Ersatzmeldung, die eine
harmlose Ursache behauptet und die echte verdeckt.

Der Lauf rechnet deshalb jetzt zwei Abdeckungen aus und spricht sie aus:

| Maß | Frage | Untergrenze |
|---|---|---|
| Datenabdeckung | Wie viele Titel lieferten überhaupt Daten? | 50 % |
| Bewertungsabdeckung | Wie viele der gefilterten Titel wurden bewertet? | 50 % |

Fällt eine darunter, gilt der Lauf als unvollständig. Dann sagen Telegram,
Konsole und HTML-Report „Lauf unvollständig — keine Aussage über den Markt
möglich" samt Zahlen, **nichts** geht ins Journal, und der Workflow endet rot.
Die Hälfte ist keine feinjustierte Zahl, sondern die Grenze, ab der „keine
Treffer" aufhört, eine Aussage über den Markt zu sein.

Der Journalschutz ist der wichtigere Teil: Das Journal ist die Lerngrundlage.
Eine Handvoll Titel, die nur deshalb Treffer wurden, weil ihre Konkurrenz an
einem Abrufproblem hängenblieb, verzerrt jede spätere Auswertung — und die
Kontrollgruppe, aus einer Rumpfmenge gezogen, misst gegen nichts.

### Die Ursache: Yahoos Zeitfenster

Der Fehlertext ist eindeutig:

```
Too Many Requests. Rate limited. Try after a while.
```

Ein Lauf über das ganze Universum braucht rund 674 Fundamentaldaten-Abrufe plus
etwa 220 Kurshistorien, zusammen gut 900. Durchgekommen sind 342 (5. August) und
334 (6. August) — beide Male etwa ein Drittel. Durch ein Fenster von 340 bekommt
man 900 Abrufe mit keinem Tempo. Nur durch das nächste Fenster.

Deshalb wartet der Lauf bei Drosselung bis zu dreimal je zwei Minuten und holt
die offenen Abrufe mit zwei statt acht Verbindungen nach. **Abgebrochen wird,
sobald eine Runde nichts zurückholt** — das ist die eigentliche Regel: Bringt
eine Pause keinen einzigen Titel zurück, ist es keine Drosselung, und weiteres
Warten schafft nur einen Lauf, der ins Zeitlimit kriecht, statt das Problem zu
melden. Die Rundenzahl begrenzt, die Erholung entscheidet.

Gewartet wird ohnehin nur, wenn der Ausfall nach Drosselung aussieht: Zwei
Minuten Pause wegen eines delisteten Titels wären verschwendet.

Dass das reicht, ist damit nicht bewiesen — am 3. August lief derselbe Code über
dasselbe Universum ohne Wiederholung durch und lieferte 15 Treffer. Wie viel
Kontingent übrig ist, hängt auch daran, wie viel andere GitHub-Runner unter
derselben Adresse schon verbraucht haben. Bleibt es dabei, ist der ehrliche
nächste Schritt kein weiteres Nachfassen, sondern ein Datenanbieter mit
zugesicherten Kontingenten.

## Wartung — die Drift-Prüfung

Die Fehler, die dieses Werkzeug im Betrieb einholen, sind selten
Programmierfehler. Es sind Änderungen draußen: Ein Index tauscht Mitglieder aus,
Eurostat benennt den Euroraum von `EA20` in `EA21` um, Yahoo ändert die
Bezeichnung einer Bilanzzeile. Jede einzelne davon ist harmlos und läuft still
ins Leere — eine Warnung im Log, die niemand liest, und eine Kennzahl, die ab
dann fehlt.

```bash
broker doctor                 # alle drei Prüfungen, Klartext
broker doctor --json          # maschinenlesbar
broker doctor --fix           # nur die mechanisch behebbaren Befunde
```

| Prüfung | Findet | Mechanisch behebbar |
|---|---|---|
| **Tote Ticker** | Titel ohne Kursdaten — Delisting, Übernahme, Börsenwechsel | ja: Zeile aus der CSV |

| **Ausgefallene Makroreihen** | erwartete Reihen, die nicht mehr ankommen | nein |
| **Umbenannte Abschlusszeilen** | Felder, die im *ganzen* Universum leer sind | nein |

Geprüft wird gegen die lebenden Schnittstellen, nicht gegen Logtext: Ein
Logformat ändert sich, sobald jemand eine Meldung umformuliert, die Frage
„antwortet diese Reihe noch?" bleibt dieselbe.

**Zwei Runden vor jedem Löschvorschlag — und die zweite fragt anders.** Der
erste Durchgang holt eine kurze Historie (`1mo`), weil 674 Abrufe sonst dauern.
Wer daraufhin nichts liefert, wird einzeln und mit Pause noch einmal gefragt,
diesmal mit **derselben Anfrage, die der Screener stellt** (`3y`).

Das ist keine Feinheit, sondern die Definition: „tot" soll heißen *der Screener
kann den Titel nicht mehr verwenden* — nicht *ein anderer, kürzerer Abruf kam
leer zurück*. Solange die Prüfung etwas anderes fragt als der Screener, misst
sie das Falsche. Antwortet ein Titel auf `3y`, landet er nicht im
Löschvorschlag, sondern als Warnung im Log.

Die zweite Runde entstand aus einer falschen Vermutung. Beim ersten scharfen
Lauf hielt ich fünf der zwanzig gemeldeten Titel für lebendig und nahm an, sie
seien kurz nicht erreichbar gewesen. Drei Läufe an drei Tagen lieferten danach
exakt dieselben zwanzig Namen — ein Wackelkontakt trifft jedes Mal andere. Die
zweite Runde blieb richtig, die Begründung war es nicht.

**Die wichtigste Sicherung ist eine Nicht-Aktion.** Fallen mehr als 20 % der
Titel gleichzeitig aus, ist nicht das Universum veraltet, sondern die
Datenquelle gestört. Diese Prüfung greift **vor** der zweiten Runde — sonst
liefen bei einem breiten Ausfall hunderte Einzelabrufe mit Pause ins Zeitlimit,
für ein Ergebnis, das ohnehin verworfen wird. Dann meldet die Prüfung eine Störung und schlägt
ausdrücklich *nichts* zur Entfernung vor — ein Pull Request, der halb Europa
aus dem Universum löscht, weil Yahoo zehn Minuten nicht antwortet, wäre das
Gefährlichste am ganzen Werkzeug.

`.github/workflows/maintenance.yml` führt das sonntags aus. Findet es nichts,
endet der Lauf still. Sonst:

* **Tote Ticker** → Pull Request gegen den Arbeitszweig, nie ein direkter Push.
  Der Text weist darauf hin, dass ein Ausfall auch eine Umbenennung sein kann —
  dann gehört das neue Kürzel eingetragen statt der Zeile gelöscht.
* **Alles andere** → ein Issue, das bei Wiederholung kommentiert statt neu
  angelegt wird.

Repariert wird aus dem bereits erstellten Bericht (`--from-report`), nicht aus
einem zweiten Abruf: Sonst stünden im Pull Request womöglich andere Titel als
die, die tatsächlich entfernt wurden.

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

361 Tests, alle mit synthetischen Daten und ohne Netzwerkzugriff.

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
