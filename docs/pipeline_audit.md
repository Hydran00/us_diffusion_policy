# Audit e correzione della pipeline XYZ

Contratto: storia di 3 immagini e 3 stati da 23 valori; USFM ViT-B/16 congelato,
MLP stato, ConditionalUnet1D e DDPM epsilon. Predizione nativa di parametri
spline XYZ per 2 s; esecuzione di 0.4 s e ripianificazione sul TCP misurato.
Il riferimento sono le specifiche testuali fornite; non è stata effettuata
una verifica bibliografica indipendente del paper.

| File / classe | CURRENT prima della patch | PROBLEM | CHANGE applicata |
|---|---|---|---|
| `dataset/convert.py`, `import_hdf5` | Importava anche solo TCP a 9 valori | Eliminava q/dq implicitamente | Richiede q e dq oppure stato completo con ordine dichiarato |
| `dataset/processing.py`, `prepare`, `WindowDataset` | Finestre future misurate nel frame TCP, split per gruppo prima delle finestre | Accettava il contratto a 9 valori | Verifica esatta dei 23 STATE_FIELDS in preparazione e caricamento |
| `dataset/processing.py`, `statistics` | Media/std sul solo train; origine esclusa dai parametri | Nessuna divergenza | Conservato |
| `common/spline.py`, `SplineCodec` | Fitting ancorato, base quadratica C1, decoder Torch a fase normalizzata | Mancava API piatta con tempi in secondi | `codec(free_params, query_times, horizon_seconds)` inserisce deterministicamente l'origine |
| `training/model.py`, `UltrasoundSplinePolicy` | USFM+MLP+U-Net; diffusione solo sui 5 vettori liberi | Diagnostica e normalizzazione non esposte | `diffusion_batch`, `normalize_params`, `denormalize_params`, contratto input 23 |
| `DDPMScheduler` | Scheduler coseno, epsilon prediction, MSE sul rumore, padding escluso dalla loss | Nessuna divergenza | Conservato |
| `training/train.py`, `train` | AdamW, scelta best su noise MSE validation | Solo noise MSE in validation | Campionamento DDPM completo e quattro metriche separate ogni epoca |
| `training/train.py`, `evaluate` | Errori posizione e fitting | Mancavano errore parametri e noise MSE | Metriche condivise con validation |
| `training/train.py`, `load_policy` | Riapriva anche il file USFM originale | Checkpoint policy non portabile da solo | Carica tutti i pesi dal checkpoint policy senza riaprire USFM |
| `deployment/inference.py`, `RecedingHorizonPolicy` | Storia reale, frame locale, prefisso temporizzato | Nessun R0 nel contratto restituito | R0 al primo observe dopo reset, orientamento costante restituito per ogni riferimento |
| `i4h-workflows/tasks/us_dp/.../server.py` | Stato 9D, triplicazione dello stesso frame, prefisso 1 s, orientamento aggiornato a ogni piano | Divergenze dalla pipeline richiesta | Storia reale, q/dq, prefisso dal checkpoint (default 0.4 s), orientamento fissato per sessione |
| `i4h-workflows/engine/i4h_engine/remote.py` | Una sola osservazione per richiesta di azioni | Non trasportava la storia durante i chunk | Raccolta locale a cadenza negoziata, invio finestra reale al replan; warmup con avanzamento simulato |
| `i4h-workflows/common/i4h_common/{server.py,bus/messages.py}` | Messaggio con q e TCP | Mancavano dq e storia | Campi opzionali compatibili: velocità, storia codificata, lunghezza/cadenza nel ready handshake |

## Parametrizzazione

4 segmenti quadratici, 3 control point XYZ per segmento: 12 vettori prima dei
vincoli. I tre raccordi C0 e i tre raccordi C1 eliminano 6 vettori, lasciandone 6.
L'origine locale f(0)=0 ne elimina un altro: **5 vettori XYZ = 15 scalari**.
La matrice C upstream ricostruisce i control point esattamente; nessuna penalità
soft di continuità. La U-Net riceve `[B,5,3]`, padded internamente a `[B,8,3]`
con la configurazione predefinita. Il padding non entra nella loss.

Il formato su disco resta `[B,6,3]` con prima riga zero per mantenere compatibilità
con i dataset 23D già preparati. La policy predice solo le restanti cinque righe.
Il decoder pubblico accetta `[B,15]`, inserisce la prima riga e campiona in Torch.
Le vecchie API `decode`/`sample` restano primitive per fitting e compatibilità;
non sono una seconda action representation.

`p_local = R_t.T @ (p_world - p_t)` nel dataset;
`p_world = R_t @ p_local + p_t` nel deployment. Il riferimento locale viene
ricostruito dal TCP misurato a ogni piano, anche se l'orientamento desiderato R0
rimane costante. Continuità C1 garantita fra segmenti dello stesso piano;
la velocità fra due ripianificazioni non è vincolata.

## Metriche

- `noise_mse`: MSE epsilon DDPM nello spazio normalizzato.
- `spline_parameter_mse_m2`: MSE sui 15 coefficienti liberi denormalizzati.
- `trajectory_mse_m2`: media temporale/batch della distanza XYZ quadratica fra
  spline predetta e traiettoria esperta misurata.
- `spline_fit_mse_m2`: stessa misura fra spline fitted e traiettoria esperta.

Validation usa prefisso `validation_`. La loss ottimizzata resta solo noise MSE;
nessuna auxiliary loss è abilitata. La selezione di `best.pt` resta su validation
noise MSE. L'errore policy rispetto all'esperto include il limite di fitting,
mentre l'errore parametri confronta la previsione con la spline fitted.
Il test conserva anche errore medio in metri e RMSE posizione.
La RMSE di fitting storica nel manifest è per coordinata (MSE XYZ divisa per 3).

## Un batch training e inference

Dalla radice workspace:

```bash
us_dp/.venv/bin/python -m us_dp.training.dry_run --pretrained USFM/USFM_latest.pth
```

Il comando usa USFM reale e osservazioni sintetiche, esegue forward/backward,
un passo AdamW e sampling DDPM. Le metriche dopo un solo batch casuale non
misurano la qualità della scansione. Report eseguito: `dry_run_example.json`.

| Tensore | Shape default, batch 2 |
|---|---|
| images | `[2,3,1,128,128]` |
| robot_state | `[2,3,23]` |
| visual_features | `[2,128]` |
| state_features | `[2,128]` |
| global_cond | `[2,256]` |
| clean_spline_params normalizzati | `[2,5,3]` |
| noisy_spline_params | `[2,5,3]` |
| predicted_noise | `[2,5,3]` |
| flat_free_params in metri | `[2,15]` |
| decoded_xyz a 10 Hz, incluso t=0 | `[2,21,3]` |
| piano completo a 50 Hz, incluso t=0 | `[2,101,3]` |
| prefisso eseguito a 50 Hz, escluso t=0 | `[2,20,3]` |

## Verifica e limiti

I test coprono ancora iniziale, raccordi C0/C1, fitting e dimensioni,
autograd del decoder, trasformazioni world/TCP, denormalizzazione non banale,
orizzonte completo 2 s e prefisso 0.4 s, R0 costante tra piani, separazione delle
metriche, rifiuto input 9D, storia remota reale e trasporto dq.

Il trasporto remoto mantiene il comportamento precedente per backend che
richiedono una sola osservazione. Per us_dp la cadenza deve essere rappresentabile
con un numero intero di tick della scena; in caso contrario rifiuta il contratto.
L'attesa del backend usa il comportamento WAITING già presente nell'engine.

Questa patch non certifica un rollout Isaac: restano da verificare in simulazione
tracking, contatto, copertura e freschezza/sincronizzazione del sensore acustico.
Il trasporto non inventa osservazioni, ma non può garantire che il renderer abbia
prodotto un B-mode nuovo a ogni tick campionato. Non sono stati aggiunti controllo
forza, collision avoidance, nuovi backbone o flow-field mode.

Verifica eseguita: **54 test us_dp** e **68 test mirati common/engine/server i4h**
passati su CPU. Dry-run con USFM_latest.pth e configurazione predefinita completato.
