# Attribution — Thingi10K training set

The 300 models this project was fine-tuned on are **not redistributed here**.
`manifest.jsonl` records which they are by Thingi10K `file_id`; each was
downloaded from Thingiverse and used under its own licence, listed below.

`training/dataset_build.py` is seeded, so the manifest plus one command
rebuilds the corpus exactly. See `training/README.md`.

## Licence breakdown

| licence | models | derived tensors redistributable? |
|---|---:|---|
| Creative Commons - Attribution - Share Alike | 102 | yes, attribution + share-alike |
| Creative Commons - Attribution | 94 | yes, with attribution |
| Creative Commons - Attribution - Non-Commercial | 45 | non-commercial only |
| Attribution - Non-Commercial - Share Alike | 40 | non-commercial only |
| Attribution - Non-Commercial - No Derivatives | 9 | NO -- no derivatives |
| GNU - GPL | 6 | yes, under GPL terms |
| Creative Commons - Attribution - No Derivatives | 2 | NO -- no derivatives |
| Creative Commons - Public Domain Dedication | 2 | yes, unrestricted |

**11 models are No-Derivatives**: rendered images and encoded latents are
derivative works, so those cannot be republished in derived form at all.
**94 are Non-Commercial.** **142 are Share-Alike**, which asks derivatives to
carry the same licence — awkward beside this repository's MIT. Shipping the
index rather than the content avoids all three problems.

## Per-model sources

| Thingi10K file | Thingiverse | licence |
|---|---|---|
| `37278` | [11654](https://www.thingiverse.com/thing:11654) | Creative Commons - Attribution - Share Alike |
| `37323` | [11629](https://www.thingiverse.com/thing:11629) | Attribution - Non-Commercial - Share Alike |
| `37841` | [11776](https://www.thingiverse.com/thing:11776) | Creative Commons - Attribution - Share Alike |
| `37881` | [11836](https://www.thingiverse.com/thing:11836) | Creative Commons - Attribution - Share Alike |
| `38297` | [11945](https://www.thingiverse.com/thing:11945) | Attribution - Non-Commercial - Share Alike |
| `39159` | [12120](https://www.thingiverse.com/thing:12120) | Creative Commons - Attribution |
| `39345` | [12180](https://www.thingiverse.com/thing:12180) | Creative Commons - Attribution - Share Alike |
| `39579` | [12256](https://www.thingiverse.com/thing:12256) | Creative Commons - Attribution |
| `39678` | [12285](https://www.thingiverse.com/thing:12285) | Creative Commons - Attribution - Share Alike |
| `40067` | [12405](https://www.thingiverse.com/thing:12405) | Creative Commons - Attribution - Share Alike |
| `42634` | [13073](https://www.thingiverse.com/thing:13073) | Creative Commons - Attribution - Non-Commercial |
| `43096` | [13143](https://www.thingiverse.com/thing:13143) | Creative Commons - Attribution |
| `43387` | [13298](https://www.thingiverse.com/thing:13298) | Creative Commons - Attribution - Share Alike |
| `43551` | [13360](https://www.thingiverse.com/thing:13360) | Creative Commons - Attribution - Non-Commercial |
| `44100` | [13522](https://www.thingiverse.com/thing:13522) | Creative Commons - Attribution - Share Alike |
| `44399` | [13531](https://www.thingiverse.com/thing:13531) | Creative Commons - Attribution - Share Alike |
| `45169` | [13753](https://www.thingiverse.com/thing:13753) | Creative Commons - Public Domain Dedication |
| `45410` | [13928](https://www.thingiverse.com/thing:13928) | Creative Commons - Attribution - Share Alike |
| `45562` | [13968](https://www.thingiverse.com/thing:13968) | Creative Commons - Attribution - Share Alike |
| `45617` | [13740](https://www.thingiverse.com/thing:13740) | Creative Commons - Attribution - Share Alike |
| `47775` | [14493](https://www.thingiverse.com/thing:14493) | Creative Commons - Attribution - Share Alike |
| `47851` | [14528](https://www.thingiverse.com/thing:14528) | Creative Commons - Attribution - Share Alike |
| `48419` | [14682](https://www.thingiverse.com/thing:14682) | Creative Commons - Attribution - Share Alike |
| `49424` | [14986](https://www.thingiverse.com/thing:14986) | Creative Commons - Attribution - Non-Commercial |
| `51639` | [15644](https://www.thingiverse.com/thing:15644) | GNU - GPL |
| `54471` | [16596](https://www.thingiverse.com/thing:16596) | Attribution - Non-Commercial - Share Alike |
| `55039` | [16775](https://www.thingiverse.com/thing:16775) | Creative Commons - Attribution - Share Alike |
| `55559` | [16954](https://www.thingiverse.com/thing:16954) | Creative Commons - Attribution - Share Alike |
| `56265` | [17204](https://www.thingiverse.com/thing:17204) | Attribution - Non-Commercial - Share Alike |
| `56498` | [15119](https://www.thingiverse.com/thing:15119) | Creative Commons - Attribution - Share Alike |
| `56524` | [17309](https://www.thingiverse.com/thing:17309) | Creative Commons - Attribution - Non-Commercial |
| `58261` | [17941](https://www.thingiverse.com/thing:17941) | Attribution - Non-Commercial - Share Alike |
| `59705` | [18466](https://www.thingiverse.com/thing:18466) | Creative Commons - Attribution - Share Alike |
| `60099` | [18610](https://www.thingiverse.com/thing:18610) | Creative Commons - Attribution |
| `60514` | [18767](https://www.thingiverse.com/thing:18767) | Creative Commons - Attribution - Share Alike |
| `61192` | [19066](https://www.thingiverse.com/thing:19066) | Creative Commons - Attribution - Share Alike |
| `64821` | [20654](https://www.thingiverse.com/thing:20654) | Creative Commons - Attribution - Share Alike |
| `64957` | [20719](https://www.thingiverse.com/thing:20719) | Creative Commons - Attribution |
| `65002` | [20739](https://www.thingiverse.com/thing:20739) | Creative Commons - Public Domain Dedication |
| `65443` | [20913](https://www.thingiverse.com/thing:20913) | Creative Commons - Attribution - Share Alike |
| `65586` | [20976](https://www.thingiverse.com/thing:20976) | Creative Commons - Attribution |
| `66773` | [21495](https://www.thingiverse.com/thing:21495) | Creative Commons - Attribution - Share Alike |
| `67408` | [21767](https://www.thingiverse.com/thing:21767) | Creative Commons - Attribution |
| `67497` | [21842](https://www.thingiverse.com/thing:21842) | Creative Commons - Attribution |
| `67516` | [21854](https://www.thingiverse.com/thing:21854) | Creative Commons - Attribution - Share Alike |
| `67550` | [21864](https://www.thingiverse.com/thing:21864) | Attribution - Non-Commercial - Share Alike |
| `67923` | [22018](https://www.thingiverse.com/thing:22018) | Creative Commons - Attribution |
| `68203` | [22134](https://www.thingiverse.com/thing:22134) | Creative Commons - Attribution - Share Alike |
| `68370` | [22203](https://www.thingiverse.com/thing:22203) | Creative Commons - Attribution |
| `68647` | [22258](https://www.thingiverse.com/thing:22258) | Creative Commons - Attribution |
| `68659` | [22258](https://www.thingiverse.com/thing:22258) | Creative Commons - Attribution |
| `68812` | [22398](https://www.thingiverse.com/thing:22398) | Creative Commons - Attribution - Share Alike |
| `68935` | [22454](https://www.thingiverse.com/thing:22454) | Creative Commons - Attribution |
| `69325` | [17773](https://www.thingiverse.com/thing:17773) | Creative Commons - Attribution - Non-Commercial |
| `69537` | [22668](https://www.thingiverse.com/thing:22668) | Creative Commons - Attribution |
| `71260` | [23370](https://www.thingiverse.com/thing:23370) | Creative Commons - Attribution - Non-Commercial |
| `71265` | [23370](https://www.thingiverse.com/thing:23370) | Creative Commons - Attribution - Non-Commercial |
| `71383` | [23437](https://www.thingiverse.com/thing:23437) | Creative Commons - Attribution |
| `72101` | [23696](https://www.thingiverse.com/thing:23696) | Creative Commons - Attribution |
| `72581` | [23906](https://www.thingiverse.com/thing:23906) | Creative Commons - Attribution |
| `72668` | [23696](https://www.thingiverse.com/thing:23696) | Creative Commons - Attribution |
| `73157` | [24152](https://www.thingiverse.com/thing:24152) | Creative Commons - Attribution - Share Alike |
| `73160` | [24152](https://www.thingiverse.com/thing:24152) | Creative Commons - Attribution - Share Alike |
| `73162` | [24152](https://www.thingiverse.com/thing:24152) | Creative Commons - Attribution - Share Alike |
| `73986` | [24501](https://www.thingiverse.com/thing:24501) | Creative Commons - Attribution - Non-Commercial |
| `74492` | [24669](https://www.thingiverse.com/thing:24669) | Attribution - Non-Commercial - Share Alike |
| `75269` | [24719](https://www.thingiverse.com/thing:24719) | Creative Commons - Attribution - Share Alike |
| `76714` | [25551](https://www.thingiverse.com/thing:25551) | Creative Commons - Attribution |
| `77011` | [25680](https://www.thingiverse.com/thing:25680) | Creative Commons - Attribution |
| `77336` | [25837](https://www.thingiverse.com/thing:25837) | Creative Commons - Attribution |
| `77340` | [25837](https://www.thingiverse.com/thing:25837) | Creative Commons - Attribution |
| `77916` | [26025](https://www.thingiverse.com/thing:26025) | Creative Commons - Attribution - Share Alike |
| `77939` | [26027](https://www.thingiverse.com/thing:26027) | Creative Commons - Attribution - No Derivatives |
| `77949` | [26027](https://www.thingiverse.com/thing:26027) | Creative Commons - Attribution - No Derivatives |
| `78251` | [26125](https://www.thingiverse.com/thing:26125) | Attribution - Non-Commercial - Share Alike |
| `78351` | [26178](https://www.thingiverse.com/thing:26178) | Creative Commons - Attribution |
| `79194` | [26519](https://www.thingiverse.com/thing:26519) | Creative Commons - Attribution |
| `79195` | [26519](https://www.thingiverse.com/thing:26519) | Creative Commons - Attribution |
| `79241` | [26536](https://www.thingiverse.com/thing:26536) | Creative Commons - Attribution |
| `79810` | [26746](https://www.thingiverse.com/thing:26746) | Creative Commons - Attribution - Non-Commercial |
| `79955` | [26811](https://www.thingiverse.com/thing:26811) | Creative Commons - Attribution |
| `80414` | [26988](https://www.thingiverse.com/thing:26988) | Creative Commons - Attribution - Non-Commercial |
| `80557` | [27050](https://www.thingiverse.com/thing:27050) | Creative Commons - Attribution |
| `82378` | [27733](https://www.thingiverse.com/thing:27733) | Creative Commons - Attribution - Share Alike |
| `82379` | [27733](https://www.thingiverse.com/thing:27733) | Creative Commons - Attribution - Share Alike |
| `84931` | [28762](https://www.thingiverse.com/thing:28762) | Creative Commons - Attribution |
| `85860` | [29115](https://www.thingiverse.com/thing:29115) | Creative Commons - Attribution - Share Alike |
| `87599` | [29832](https://www.thingiverse.com/thing:29832) | Creative Commons - Attribution - Share Alike |
| `87688` | [29876](https://www.thingiverse.com/thing:29876) | Attribution - Non-Commercial - No Derivatives |
| `87721` | [29860](https://www.thingiverse.com/thing:29860) | Creative Commons - Attribution |
| `90207` | [30952](https://www.thingiverse.com/thing:30952) | GNU - GPL |
| `90275` | [30981](https://www.thingiverse.com/thing:30981) | Creative Commons - Attribution |
| `91606` | [31497](https://www.thingiverse.com/thing:31497) | Creative Commons - Attribution - Share Alike |
| `92668` | [31944](https://www.thingiverse.com/thing:31944) | Attribution - Non-Commercial - Share Alike |
| `94016` | [31392](https://www.thingiverse.com/thing:31392) | Creative Commons - Attribution - Share Alike |
| `94674` | [32053](https://www.thingiverse.com/thing:32053) | Attribution - Non-Commercial - Share Alike |
| `94733` | [32773](https://www.thingiverse.com/thing:32773) | Creative Commons - Attribution - Share Alike |
| `95433` | [33048](https://www.thingiverse.com/thing:33048) | Creative Commons - Attribution - Share Alike |
| `95487` | [33093](https://www.thingiverse.com/thing:33093) | Creative Commons - Attribution |
| `95494` | [33096](https://www.thingiverse.com/thing:33096) | Creative Commons - Attribution |
| `95498` | [33096](https://www.thingiverse.com/thing:33096) | Creative Commons - Attribution |
| `96660` | [33511](https://www.thingiverse.com/thing:33511) | Creative Commons - Attribution |
| `97939` | [33983](https://www.thingiverse.com/thing:33983) | Creative Commons - Attribution |
| `98019` | [33988](https://www.thingiverse.com/thing:33988) | Creative Commons - Attribution |
| `98663` | [34295](https://www.thingiverse.com/thing:34295) | Creative Commons - Attribution |
| `100339` | [34904](https://www.thingiverse.com/thing:34904) | Creative Commons - Attribution |
| `100478` | [35012](https://www.thingiverse.com/thing:35012) | Creative Commons - Attribution |
| `100642` | [34772](https://www.thingiverse.com/thing:34772) | Creative Commons - Attribution |
| `101634` | [35437](https://www.thingiverse.com/thing:35437) | Attribution - Non-Commercial - Share Alike |
| `103141` | [36059](https://www.thingiverse.com/thing:36059) | Creative Commons - Attribution - Share Alike |
| `103537` | [36200](https://www.thingiverse.com/thing:36200) | Creative Commons - Attribution - Share Alike |
| `104442` | [36200](https://www.thingiverse.com/thing:36200) | Creative Commons - Attribution - Share Alike |
| `105338` | [36958](https://www.thingiverse.com/thing:36958) | Attribution - Non-Commercial - Share Alike |
| `105686` | [37084](https://www.thingiverse.com/thing:37084) | Creative Commons - Attribution - Share Alike |
| `105924` | [36554](https://www.thingiverse.com/thing:36554) | Creative Commons - Attribution |
| `107389` | [37727](https://www.thingiverse.com/thing:37727) | Creative Commons - Attribution - Share Alike |
| `107402` | [37727](https://www.thingiverse.com/thing:37727) | Creative Commons - Attribution - Share Alike |
| `107587` | [36554](https://www.thingiverse.com/thing:36554) | Creative Commons - Attribution |
| `109375` | [36967](https://www.thingiverse.com/thing:36967) | Creative Commons - Attribution - Share Alike |
| `110786` | [38990](https://www.thingiverse.com/thing:38990) | Attribution - Non-Commercial - Share Alike |
| `110796` | [36554](https://www.thingiverse.com/thing:36554) | Creative Commons - Attribution |
| `110909` | [39045](https://www.thingiverse.com/thing:39045) | Creative Commons - Attribution - Share Alike |
| `111006` | [38990](https://www.thingiverse.com/thing:38990) | Attribution - Non-Commercial - Share Alike |
| `111021` | [39089](https://www.thingiverse.com/thing:39089) | Creative Commons - Attribution - Non-Commercial |
| `112919` | [39845](https://www.thingiverse.com/thing:39845) | GNU - GPL |
| `112940` | [39845](https://www.thingiverse.com/thing:39845) | GNU - GPL |
| `113866` | [40190](https://www.thingiverse.com/thing:40190) | Attribution - Non-Commercial - Share Alike |
| `113887` | [39368](https://www.thingiverse.com/thing:39368) | Creative Commons - Attribution - Non-Commercial |
| `115053` | [40634](https://www.thingiverse.com/thing:40634) | Attribution - Non-Commercial - No Derivatives |
| `116051` | [41201](https://www.thingiverse.com/thing:41201) | Creative Commons - Attribution - Share Alike |
| `117525` | [42027](https://www.thingiverse.com/thing:42027) | Creative Commons - Attribution |
| `118295` | [42265](https://www.thingiverse.com/thing:42265) | Attribution - Non-Commercial - No Derivatives |
| `121868` | [44185](https://www.thingiverse.com/thing:44185) | Creative Commons - Attribution - Share Alike |
| `123958` | [45203](https://www.thingiverse.com/thing:45203) | Creative Commons - Attribution - Share Alike |
| `124036` | [42877](https://www.thingiverse.com/thing:42877) | Creative Commons - Attribution |
| `124373` | [45347](https://www.thingiverse.com/thing:45347) | Creative Commons - Attribution |
| `129891` | [48377](https://www.thingiverse.com/thing:48377) | Creative Commons - Attribution |
| `129906` | [48377](https://www.thingiverse.com/thing:48377) | Creative Commons - Attribution |
| `129909` | [48377](https://www.thingiverse.com/thing:48377) | Creative Commons - Attribution |
| `129911` | [48377](https://www.thingiverse.com/thing:48377) | Creative Commons - Attribution |
| `129917` | [48377](https://www.thingiverse.com/thing:48377) | Creative Commons - Attribution |
| `130046` | [48418](https://www.thingiverse.com/thing:48418) | Creative Commons - Attribution - Share Alike |
| `130788` | [48912](https://www.thingiverse.com/thing:48912) | Attribution - Non-Commercial - Share Alike |
| `130980` | [49033](https://www.thingiverse.com/thing:49033) | Creative Commons - Attribution - Share Alike |
| `131453` | [49348](https://www.thingiverse.com/thing:49348) | Creative Commons - Attribution - Share Alike |
| `131725` | [49477](https://www.thingiverse.com/thing:49477) | Creative Commons - Attribution - Share Alike |
| `132349` | [49834](https://www.thingiverse.com/thing:49834) | Creative Commons - Attribution - Non-Commercial |
| `132425` | [49865](https://www.thingiverse.com/thing:49865) | Creative Commons - Attribution |
| `133079` | [50207](https://www.thingiverse.com/thing:50207) | Creative Commons - Attribution - Non-Commercial |
| `134543` | [50319](https://www.thingiverse.com/thing:50319) | Creative Commons - Attribution - Share Alike |
| `135065` | [51112](https://www.thingiverse.com/thing:51112) | Creative Commons - Attribution - Share Alike |
| `135074` | [51112](https://www.thingiverse.com/thing:51112) | Creative Commons - Attribution - Share Alike |
| `136236` | [51829](https://www.thingiverse.com/thing:51829) | Attribution - Non-Commercial - No Derivatives |
| `136935` | [51489](https://www.thingiverse.com/thing:51489) | Attribution - Non-Commercial - Share Alike |
| `145331` | [55553](https://www.thingiverse.com/thing:55553) | Creative Commons - Attribution - Share Alike |
| `161091` | [65740](https://www.thingiverse.com/thing:65740) | Creative Commons - Attribution - Non-Commercial |
| `178193` | [76369](https://www.thingiverse.com/thing:76369) | Creative Commons - Attribution |
| `186544` | [81830](https://www.thingiverse.com/thing:81830) | Attribution - Non-Commercial - Share Alike |
| `190247` | [84142](https://www.thingiverse.com/thing:84142) | Creative Commons - Attribution |
| `190249` | [84142](https://www.thingiverse.com/thing:84142) | Creative Commons - Attribution |
| `196126` | [86982](https://www.thingiverse.com/thing:86982) | Creative Commons - Attribution |
| `196189` | [86982](https://www.thingiverse.com/thing:86982) | Creative Commons - Attribution |
| `196193` | [86982](https://www.thingiverse.com/thing:86982) | Creative Commons - Attribution |
| `200967` | [90302](https://www.thingiverse.com/thing:90302) | Attribution - Non-Commercial - No Derivatives |
| `204954` | [93320](https://www.thingiverse.com/thing:93320) | Creative Commons - Attribution |
| `205450` | [93657](https://www.thingiverse.com/thing:93657) | Creative Commons - Attribution - Non-Commercial |
| `206495` | [86982](https://www.thingiverse.com/thing:86982) | Creative Commons - Attribution |
| `212133` | [97870](https://www.thingiverse.com/thing:97870) | Attribution - Non-Commercial - Share Alike |
| `212138` | [97870](https://www.thingiverse.com/thing:97870) | Attribution - Non-Commercial - Share Alike |
| `225955` | [106595](https://www.thingiverse.com/thing:106595) | Creative Commons - Attribution |
| `225971` | [106595](https://www.thingiverse.com/thing:106595) | Creative Commons - Attribution |
| `229601` | [108834](https://www.thingiverse.com/thing:108834) | Creative Commons - Attribution |
| `237624` | [113865](https://www.thingiverse.com/thing:113865) | Creative Commons - Attribution - Share Alike |
| `237634` | [113865](https://www.thingiverse.com/thing:113865) | Creative Commons - Attribution - Share Alike |
| `237739` | [113908](https://www.thingiverse.com/thing:113908) | Creative Commons - Attribution |
| `248185` | [120128](https://www.thingiverse.com/thing:120128) | Creative Commons - Attribution - Non-Commercial |
| `248186` | [120128](https://www.thingiverse.com/thing:120128) | Creative Commons - Attribution - Non-Commercial |
| `248191` | [120128](https://www.thingiverse.com/thing:120128) | Creative Commons - Attribution - Non-Commercial |
| `252636` | [123231](https://www.thingiverse.com/thing:123231) | Creative Commons - Attribution - Non-Commercial |
| `269127` | [134247](https://www.thingiverse.com/thing:134247) | Creative Commons - Attribution - Share Alike |
| `271868` | [136235](https://www.thingiverse.com/thing:136235) | Creative Commons - Attribution |
| `271873` | [136235](https://www.thingiverse.com/thing:136235) | Creative Commons - Attribution |
| `276936` | [135718](https://www.thingiverse.com/thing:135718) | GNU - GPL |
| `280281` | [142350](https://www.thingiverse.com/thing:142350) | Creative Commons - Attribution - Non-Commercial |
| `286158` | [146400](https://www.thingiverse.com/thing:146400) | Creative Commons - Attribution - Share Alike |
| `289654` | [149026](https://www.thingiverse.com/thing:149026) | Creative Commons - Attribution - Share Alike |
| `289662` | [149026](https://www.thingiverse.com/thing:149026) | Creative Commons - Attribution - Share Alike |
| `289667` | [149026](https://www.thingiverse.com/thing:149026) | Creative Commons - Attribution - Share Alike |
| `319831` | [167043](https://www.thingiverse.com/thing:167043) | Creative Commons - Attribution - Share Alike |
| `338509` | [179226](https://www.thingiverse.com/thing:179226) | Creative Commons - Attribution - Share Alike |
| `343547` | [149026](https://www.thingiverse.com/thing:149026) | Creative Commons - Attribution - Share Alike |
| `365170` | [194864](https://www.thingiverse.com/thing:194864) | Creative Commons - Attribution - Non-Commercial |
| `370887` | [185125](https://www.thingiverse.com/thing:185125) | Attribution - Non-Commercial - Share Alike |
| `375245` | [113117](https://www.thingiverse.com/thing:113117) | Creative Commons - Attribution |
| `375264` | [113117](https://www.thingiverse.com/thing:113117) | Creative Commons - Attribution |
| `375275` | [113117](https://www.thingiverse.com/thing:113117) | Creative Commons - Attribution |
| `389251` | [113117](https://www.thingiverse.com/thing:113117) | Creative Commons - Attribution |
| `399560` | [185912](https://www.thingiverse.com/thing:185912) | Creative Commons - Attribution - Non-Commercial |
| `454346` | [248009](https://www.thingiverse.com/thing:248009) | Creative Commons - Attribution |
| `462514` | [252162](https://www.thingiverse.com/thing:252162) | Creative Commons - Attribution - Share Alike |
| `462515` | [252162](https://www.thingiverse.com/thing:252162) | Creative Commons - Attribution - Share Alike |
| `462519` | [252162](https://www.thingiverse.com/thing:252162) | Creative Commons - Attribution - Share Alike |
| `462523` | [252162](https://www.thingiverse.com/thing:252162) | Creative Commons - Attribution - Share Alike |
| `462540` | [252162](https://www.thingiverse.com/thing:252162) | Creative Commons - Attribution - Share Alike |
| `462544` | [252162](https://www.thingiverse.com/thing:252162) | Creative Commons - Attribution - Share Alike |
| `472025` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `472029` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `472039` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `472075` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `472080` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `472092` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `472136` | [259005](https://www.thingiverse.com/thing:259005) | Creative Commons - Attribution |
| `496389` | [274396](https://www.thingiverse.com/thing:274396) | Creative Commons - Attribution - Non-Commercial |
| `500089` | [276836](https://www.thingiverse.com/thing:276836) | Creative Commons - Attribution |
| `500090` | [276836](https://www.thingiverse.com/thing:276836) | Creative Commons - Attribution |
| `500095` | [276836](https://www.thingiverse.com/thing:276836) | Creative Commons - Attribution |
| `500108` | [276836](https://www.thingiverse.com/thing:276836) | Creative Commons - Attribution |
| `514853` | [284482](https://www.thingiverse.com/thing:284482) | Creative Commons - Attribution - Non-Commercial |
| `518031` | [288447](https://www.thingiverse.com/thing:288447) | Attribution - Non-Commercial - Share Alike |
| `518038` | [288447](https://www.thingiverse.com/thing:288447) | Attribution - Non-Commercial - Share Alike |
| `518087` | [288447](https://www.thingiverse.com/thing:288447) | Attribution - Non-Commercial - Share Alike |
| `518094` | [288447](https://www.thingiverse.com/thing:288447) | Attribution - Non-Commercial - Share Alike |
| `520647` | [276836](https://www.thingiverse.com/thing:276836) | Creative Commons - Attribution |
| `543706` | [300113](https://www.thingiverse.com/thing:300113) | Attribution - Non-Commercial - No Derivatives |
| `550198` | [232248](https://www.thingiverse.com/thing:232248) | Attribution - Non-Commercial - Share Alike |
| `567031` | [321624](https://www.thingiverse.com/thing:321624) | Creative Commons - Attribution - Share Alike |
| `567033` | [321624](https://www.thingiverse.com/thing:321624) | Creative Commons - Attribution - Share Alike |
| `567041` | [321624](https://www.thingiverse.com/thing:321624) | Creative Commons - Attribution - Share Alike |
| `579369` | [321827](https://www.thingiverse.com/thing:321827) | Creative Commons - Attribution - Non-Commercial |
| `593382` | [279763](https://www.thingiverse.com/thing:279763) | Attribution - Non-Commercial - No Derivatives |
| `595269` | [310961](https://www.thingiverse.com/thing:310961) | Creative Commons - Attribution - Share Alike |
| `619534` | [356580](https://www.thingiverse.com/thing:356580) | Attribution - Non-Commercial - Share Alike |
| `627212` | [356580](https://www.thingiverse.com/thing:356580) | Attribution - Non-Commercial - Share Alike |
| `636795` | [367262](https://www.thingiverse.com/thing:367262) | Creative Commons - Attribution - Share Alike |
| `636797` | [367262](https://www.thingiverse.com/thing:367262) | Creative Commons - Attribution - Share Alike |
| `636802` | [367262](https://www.thingiverse.com/thing:367262) | Creative Commons - Attribution - Share Alike |
| `636807` | [367262](https://www.thingiverse.com/thing:367262) | Creative Commons - Attribution - Share Alike |
| `641139` | [356580](https://www.thingiverse.com/thing:356580) | Attribution - Non-Commercial - Share Alike |
| `669967` | [392115](https://www.thingiverse.com/thing:392115) | Creative Commons - Attribution |
| `682291` | [376158](https://www.thingiverse.com/thing:376158) | Creative Commons - Attribution - Share Alike |
| `697192` | [380665](https://www.thingiverse.com/thing:380665) | Attribution - Non-Commercial - Share Alike |
| `697222` | [242639](https://www.thingiverse.com/thing:242639) | Attribution - Non-Commercial - Share Alike |
| `697223` | [242639](https://www.thingiverse.com/thing:242639) | Attribution - Non-Commercial - Share Alike |
| `700903` | [33357](https://www.thingiverse.com/thing:33357) | Creative Commons - Attribution - Share Alike |
| `723893` | [430050](https://www.thingiverse.com/thing:430050) | Attribution - Non-Commercial - Share Alike |
| `762595` | [456460](https://www.thingiverse.com/thing:456460) | Attribution - Non-Commercial - No Derivatives |
| `777037` | [466723](https://www.thingiverse.com/thing:466723) | Creative Commons - Attribution - Non-Commercial |
| `779996` | [468872](https://www.thingiverse.com/thing:468872) | Attribution - Non-Commercial - Share Alike |
| `780047` | [468872](https://www.thingiverse.com/thing:468872) | Attribution - Non-Commercial - Share Alike |
| `799439` | [482307](https://www.thingiverse.com/thing:482307) | Creative Commons - Attribution - Share Alike |
| `815477` | [493016](https://www.thingiverse.com/thing:493016) | Creative Commons - Attribution - Non-Commercial |
| `815484` | [493016](https://www.thingiverse.com/thing:493016) | Creative Commons - Attribution - Non-Commercial |
| `815486` | [493016](https://www.thingiverse.com/thing:493016) | Creative Commons - Attribution - Non-Commercial |
| `827641` | [442571](https://www.thingiverse.com/thing:442571) | Attribution - Non-Commercial - Share Alike |
| `827772` | [442571](https://www.thingiverse.com/thing:442571) | Attribution - Non-Commercial - Share Alike |
| `839724` | [30654](https://www.thingiverse.com/thing:30654) | Creative Commons - Attribution - Non-Commercial |
| `849727` | [43278](https://www.thingiverse.com/thing:43278) | Creative Commons - Attribution - Share Alike |
| `919986` | [570797](https://www.thingiverse.com/thing:570797) | Creative Commons - Attribution - Share Alike |
| `919992` | [570797](https://www.thingiverse.com/thing:570797) | Creative Commons - Attribution - Share Alike |
| `921796` | [511668](https://www.thingiverse.com/thing:511668) | Creative Commons - Attribution - Non-Commercial |
| `956446` | [380665](https://www.thingiverse.com/thing:380665) | Attribution - Non-Commercial - Share Alike |
| `956669` | [14826](https://www.thingiverse.com/thing:14826) | Creative Commons - Attribution |
| `1018273` | [633436](https://www.thingiverse.com/thing:633436) | Creative Commons - Attribution - Share Alike |
| `1036394` | [644933](https://www.thingiverse.com/thing:644933) | Creative Commons - Attribution - Non-Commercial |
| `1036399` | [644933](https://www.thingiverse.com/thing:644933) | Creative Commons - Attribution - Non-Commercial |
| `1063855` | [662115](https://www.thingiverse.com/thing:662115) | Creative Commons - Attribution - Non-Commercial |
| `1063860` | [662115](https://www.thingiverse.com/thing:662115) | Creative Commons - Attribution - Non-Commercial |
| `1063862` | [662115](https://www.thingiverse.com/thing:662115) | Creative Commons - Attribution - Non-Commercial |
| `1071710` | [649284](https://www.thingiverse.com/thing:649284) | Creative Commons - Attribution - Share Alike |
| `1075458` | [570797](https://www.thingiverse.com/thing:570797) | Creative Commons - Attribution - Share Alike |
| `1099667` | [684376](https://www.thingiverse.com/thing:684376) | Creative Commons - Attribution - Share Alike |
| `1120775` | [697480](https://www.thingiverse.com/thing:697480) | Creative Commons - Attribution - Share Alike |
| `1120777` | [697480](https://www.thingiverse.com/thing:697480) | Creative Commons - Attribution - Share Alike |
| `1129079` | [697635](https://www.thingiverse.com/thing:697635) | Attribution - Non-Commercial - Share Alike |
| `1130078` | [703254](https://www.thingiverse.com/thing:703254) | Creative Commons - Attribution - Share Alike |
| `1146187` | [713815](https://www.thingiverse.com/thing:713815) | Creative Commons - Attribution - Non-Commercial |
| `1255206` | [784951](https://www.thingiverse.com/thing:784951) | Creative Commons - Attribution - Share Alike |
| `1313533` | [811450](https://www.thingiverse.com/thing:811450) | Attribution - Non-Commercial - No Derivatives |
| `1315830` | [801279](https://www.thingiverse.com/thing:801279) | Creative Commons - Attribution - Non-Commercial |
| `1315831` | [801279](https://www.thingiverse.com/thing:801279) | Creative Commons - Attribution - Non-Commercial |
| `1315845` | [801279](https://www.thingiverse.com/thing:801279) | Creative Commons - Attribution - Non-Commercial |
| `1322465` | [830333](https://www.thingiverse.com/thing:830333) | Attribution - Non-Commercial - Share Alike |
| `1341742` | [843800](https://www.thingiverse.com/thing:843800) | Creative Commons - Attribution |
| `1356634` | [854906](https://www.thingiverse.com/thing:854906) | Creative Commons - Attribution |
| `1356637` | [854906](https://www.thingiverse.com/thing:854906) | Creative Commons - Attribution |
| `1378611` | [801279](https://www.thingiverse.com/thing:801279) | Creative Commons - Attribution - Non-Commercial |
| `1454018` | [401545](https://www.thingiverse.com/thing:401545) | Creative Commons - Attribution - Non-Commercial |
| `1455630` | [911205](https://www.thingiverse.com/thing:911205) | GNU - GPL |
| `1458669` | [83024](https://www.thingiverse.com/thing:83024) | Creative Commons - Attribution |
| `1458674` | [83024](https://www.thingiverse.com/thing:83024) | Creative Commons - Attribution |
| `1458682` | [83024](https://www.thingiverse.com/thing:83024) | Creative Commons - Attribution |
| `1458701` | [83024](https://www.thingiverse.com/thing:83024) | Creative Commons - Attribution |
| `1490844` | [938561](https://www.thingiverse.com/thing:938561) | Creative Commons - Attribution - Non-Commercial |
| `1505025` | [952564](https://www.thingiverse.com/thing:952564) | Creative Commons - Attribution - Share Alike |
| `1582417` | [1002335](https://www.thingiverse.com/thing:1002335) | Creative Commons - Attribution - Share Alike |
| `1582438` | [1002335](https://www.thingiverse.com/thing:1002335) | Creative Commons - Attribution - Share Alike |
| `1592665` | [1009253](https://www.thingiverse.com/thing:1009253) | Creative Commons - Attribution - Share Alike |
| `1620057` | [1015238](https://www.thingiverse.com/thing:1015238) | Creative Commons - Attribution - Non-Commercial |
| `1706466` | [1085472](https://www.thingiverse.com/thing:1085472) | Creative Commons - Attribution - Share Alike |
| `1743321` | [1068443](https://www.thingiverse.com/thing:1068443) | Creative Commons - Attribution |
