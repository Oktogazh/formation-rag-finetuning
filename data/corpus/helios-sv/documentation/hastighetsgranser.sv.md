---
id: doc-hastighetsgranser
titre: Hastighetsgränser och kvoter
source: helios/docs/sv
date: 2026-05-12
langue: sv
---

## Hastighetsgränser

Hastighetsgränsen är satt till 60 förfrågningar per minut för abonnemanget Bas.
Abonnemanget Pro tillåter 240 förfrågningar per minut, och abonnemanget Företag
1200. Gränsen räknas per miljö och inte per konto.

Varje svar innehåller en rubrik som visar återstående hastighet. Om du
överskrider gränsen returnerar tjänsten kod 429. Vänta den tid som anges innan
du försöker igen.

## Kvoter

Du kan skapa upp till 10 miljöer per abonnemang. Varje miljö har sin egen
uppsättning nycklar, och en nyckel fungerar aldrig utanför sin miljö.
Testslutpunkten förbrukar inte din kvot.

Slutpunkten för version 1 tas bort om 12 månader. Migrera dina integrationer
innan dess. Vi meddelar varje ändring 30 dagar i förväg.

## Webhooks

En webhook försöks om 5 gånger innan den överges. Kontrollera att
webhook-adressen svarar inom 3 sekunder. Ett svar som dröjer räknas som ett
misslyckande.

Uppgifterna krypteras i vila och under överföring. Listan över underleverantörer
publiceras på sekretesssidan.
