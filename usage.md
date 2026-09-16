### Collect demonstrations
```
export DISPLAY=:2.0
i4h-workflows/run.sh ultrasound_liver_scan --rule-based --episodes 200 --attempts 1   --record data/raw/demos.hdf5 --run-dir runs/ultrasound_liver_scan/acq_b   --ultrasound --record-failures --headless
```


### Prepare data 
```
us_dp/.venv/bin/us-dp prepare \
  --input runs/ultrasound_liver_scan/acq_c \
  --output data/prepared_acq_c \
  --config runs/ultrasound_liver_scan/acq_c/train_config.json
```

### Train
```
us_dp/.venv/bin/us-dp train --dataset data/prepared_acq_c --output runs/train_acq_c --device cuda:0
```



## TensorBoard

Ogni nuovo training scrive automaticamente gli eventi in `<output>/tensorboard`.
Installare le dipendenze aggiornate e avviare il viewer dalla root del workspace:

```bash
us_dp/.venv/bin/python -m pip install -e us_dp
us_dp/.venv/bin/tensorboard --logdir runs --port 6006
```

Aprire http://localhost:6006. Se il training gira su un server remoto, dal proprio
computer aprire un tunnel `ssh -L 6006:localhost:6006 USER@SERVER`.

- `step/*`: loss per batch, norma del gradiente prima del clipping e learning rate;
  l'asse orizzontale conta gli aggiornamenti del modello.
- `loss/*`: loss media train/validation e migliore validation, per epoca.
- `validation/*`: errori sui coefficienti spline, traiettoria e posizione, per epoca.
- `timing/epoch_seconds`: durata delle epoche; `config` nella scheda Text contiene la configurazione.

Gli eventi vengono scaricati su disco ogni 10 secondi e alla fine di ogni epoca.
Le metriche di validazione compaiono dopo la validazione. Un training già avviato
non acquisisce automaticamente questo logging; i nuovi eventi partono dalla prossima esecuzione.


### Test Policy
```
US_DP_SINGLE_PLAN=1 DISPLAY=:2.0 \
i4h-workflows/run.sh ultrasound_liver_scan \
  --policy \
  --task-id us_dp/ultrasound_liver_scan \
  --checkpoint /home/davide.nardi-1/i4h_ws/runs/train_1_c/best.pt \
  --episodes 1 --attempts 1 \
  --ultrasound \
  --record verify.hdf5 --record-failures
```
## Stabilità del sampling DDPM

Il sampling limita la stima dei coefficienti puliti normalizzati (`x0`) a ogni
passo DDPM mediante `sampling_clip_range`, default `4.0`. Il limite è espresso
in unità standardizzate, non in metri, e viene applicato prima del calcolo del
passo inverso. Nel dataset `data/prepared_pose` il massimo assoluto del solo
training è 3.554: ±4 include tutti i coefficienti target osservati. Su altri
dataset verificare la copertura prima di scegliere il limite.

Il caricamento dei vecchi checkpoint privi di questo campo usa il nuovo default
e stampa un avviso: i pesi restano identici, ma le vecchie metriche di sampling
non sono direttamente confrontabili. Riavviare il server policy per applicare
la modifica. Nei nuovi checkpoint il valore è salvato nella configurazione;
`evaluate` lo riporta insieme alle metriche. Impostare il campo a `null` in una
configurazione riproduce il sampling senza limiti solo per diagnostica.

Non serve riaddestrare per applicare questa correzione ai checkpoint esistenti.
Un training già in esecuzione continuerà a usare il codice caricato all'avvio.
Il limite sui coefficienti non garantisce limiti di velocità, accelerazione,
contatto o workspace del robot.

Per ripetere il confronto completo, dalla root del workspace:

```bash
us_dp/.venv/bin/python runs/diagnostics/validate_sampling_fix.py
```

Il report è `runs/diagnostics/sampling_fix_validation.json`. La modalità
`US_DP_SINGLE_PLAN=1` resta disponibile per verificare una sola traiettoria
completa di 1,0 secondo (50 step a 50 Hz) senza replanning.

## Preparazione diretta di una run i4h

`prepare --input` accetta una directory run con `run.json`, un HDF5 canonico i4h,
o una directory raw NPZ (`--raw` rimane un alias). Per HDF5 seleziona gli episodi
con `success=True`, converte immagini RGB in grayscale e usa le pose TCP misurate
in formato XYZ + quaternion WXYZ e i timestamp registrati. Il raggruppamento
usa `group_id` quando presente, altrimenti la posa iniziale del phantom arrotondata
a 5 decimali. I raw intermedi sono temporanei e vengono rimossi automaticamente.
Il manifest conserva mapping, sorgente e conteggi di selezione. Il flag di
successo non garantisce il contatto continuo. Gli episodi riusciti troppo corti
per l'orizzonte richiesto producono un errore esplicito.

## Traiettoria 3D del test single-plan

Con `US_DP_SINGLE_PLAN=1`, il backend salva automaticamente in
`<run-dir>/trajectories/` un PNG Matplotlib 3D e un NPZ con i waypoint predetti.
Gli assi world sono in metri e hanno la stessa scala; il colore indica il tempo.
Il punto iniziale è il TCP misurato, la curva è la previsione, non il moto reale.
Per aprire il file con rotazione e zoom:

```bash
DISPLAY=:2.0 us_dp/.venv/bin/python -m us_dp.deployment.plot_trajectory \
  <run-dir>/trajectories/episode_0000_step_000002.npz
```

Il nome esatto viene indicato in `backend-us_dp.log` (PNG e NPZ hanno lo stesso
nome base). Salvare la figura non richiede display e non attende la chiusura di
una finestra prima di inviare il piano.

## Ablation: immagini oscurate durante il rollout

Aggiungere `US_DP_ZERO_IMAGES=1` al comando di test. Il server passa immagini
uint8 tutte a zero all'encoder per ogni frame della storia, mantenendo pose e
timestamp reali. Le immagini registrate e quelle del simulatore restano originali.
Funziona sia con `US_DP_SINGLE_PLAN=1` sia con il replanning; senza il flag il
comportamento è invariato. `backend-us_dp.log` riporta `zero_images=True`.

```bash
US_DP_ZERO_IMAGES=1 US_DP_SINGLE_PLAN=1 DISPLAY=:2.0 \
i4h-workflows/run.sh ultrasound_liver_scan \
  --policy --task-id us_dp/ultrasound_liver_scan \
  --checkpoint /home/davide.nardi-1/i4h_ws/runs/train_1_c/best.pt \
  --episodes 1 --attempts 1 --ultrasound \
  --record verify.hdf5 --record-failures
```

Questa è un'ablation dell'input a inferenza, senza riaddestramento; l'encoder
rimane attivo e può produrre feature non nulle anche da un'immagine nera.

## Ablation: riaddestramento senza conditioning visivo

```bash
us_dp/.venv/bin/us-dp train \
  --dataset data/prepared_acq_c \
  --output runs/train_acq_c_no_image \
  --device cuda:0 \
  --no-image-conditioning
```

Il flag salva `use_image_conditioning=false` nella configurazione del checkpoint.
L'encoder USFM non viene costruito né eseguito e non richiede pesi pretrained;
la metà visiva del conditioning è un vettore nullo. La rete delle pose e il
U-Net mantengono dimensioni e obiettivo del modello completo. La pipeline dati
continua a leggere le immagini, ma i loro valori non contribuiscono al modello.
Il dataset preparato e gli split possono essere riutilizzati senza conversione.

Per valutare nel simulatore, usare il checkpoint
`runs/train_acq_c_no_image/best.pt` nello stesso comando di rollout: la modalità
senza immagini viene ricostruita automaticamente, senza `US_DP_ZERO_IMAGES`.
Mantenere `--ultrasound` perché il contratto del server richiede ancora la camera.
I vecchi checkpoint senza il campo mantengono il conditioning visivo abilitato.
