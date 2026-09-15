# Ultrasound-Guided Diffusion Spline Policy in Isaac Lab

## Replay HDF5: ultrasound e traiettoria 3D

Dalla directory `us_dp`:

```bash
.venv/bin/us-dp view-hdf5
```

Il default apre `../runs/ultrasound_liver_scan/acq_001/data/raw/demos.hdf5`
e `data/assets/abdphantom/Skin.obj`. Serve un desktop/display grafico e le
dipendenze `pip install -e '.[viewer]'`. Implementazione:
`src/us_dp/dataset_generation/hdf5_viewer.py`.

```bash
# Percorsi alternativi e frequenza di fallback senza timestamp:
.venv/bin/us-dp view-hdf5 --input /path/to/demos.hdf5 --skin /path/to/Skin.obj --sample-hz 50
# Entry point diretto equivalente:
.venv/bin/python -m us_dp.dataset_generation.hdf5_viewer
```

La mesh viene caricata direttamente da `Skin.obj` a risoluzione completa,
senza downsampling e senza cache di visualizzazione.

La finestra Open3D affianca pelle semitrasparente, traiettoria TCP misurata
progressiva e frame ultrasound registrato. Il riferimento colorato indica
posizione e orientamento TCP (`obs/measured_ee_pose`, quaternion **wxyz**), in
metri nel mondo. Le immagini vengono lette su richiesta, senza caricare l'intero
HDF5 in RAM. Il numero dell'episodio, lo stato (anche `failed`) e il tempo sono visibili.

| Tasto | Azione |
| --- | --- |
| `N` / `P` | Episodio successivo / precedente, con ritorno circolare |
| `Spazio` | Pausa / ripresa |
| `←` / `→` | Campione precedente / successivo e pausa |
| `R` | Riparti dall'inizio |
| `Home` | Reimposta camera 3D |
| `Q` | Chiudi |

Il replay si ferma alla fine dell'episodio. Il clock monotono mantiene la velocità
1× del tempo simulato; se il rendering è lento salta campioni di visualizzazione,
mantenendo immagine e traiettoria sullo stesso campione. Non interpola immagini.
Usa `obs/timestamps` se presente, altrimenti `--sample-hz`: per `acq_001` i 50 Hz
sono ricavati dal log (`sim.dt=0.005`, `decimation=4`), non da timestamp salvati.

Le nuove acquisizioni `panda_phantom` salvano inoltre, per ogni campione:

| Dataset in `obs/` | Contenuto |
| --- | --- |
| `phantom_pose` | Posa root del phantom nel mondo, `(T,7)` |
| `mesh_pose` | Posa calibrata della mesh acustica nel mondo, `(T,7)` |
| `ultrasound_probe_pose` | Posa del frame acustico della sonda nel mondo, `(T,7)` |
| `timestamps` | Tempo simulato in secondi, `(T,)` |

Le pose usano posizione xyz in metri e quaternion wxyz, dichiarati anche negli
attributi HDF5. Il viewer usa automaticamente `mesh_pose` e aggiorna la mesh a
ogni frame; considera anche gli eventuali movimenti del phantom durante l'episodio.
Il comando di acquisizione non cambia. I dati preesistenti non vengono retrocompilati:
solo i nuovi episodi contengono queste informazioni. Non si stimano pose dai waypoint.

**Limite dell'acquisizione esistente:** `acq_001` non salva la posa del phantom
randomizzata per episodio, né il frame ID/timestamp proprio del sensore ultrasound.
Il replay sincronizza le righe registrate, ma non può ricostruire la reale età di
immagini eventualmente duplicate. Negli episodi privi di posa il viewer mostra la pelle nella posa nominale,
indicandola come **NOMINAL**: non è un allineamento verificato per quell’episodio. All'apertura seleziona il primo episodio con `obs/mesh_pose`,
se presente; `N` e `P` permettono comunque di visitare tutti gli episodi.
Le unità OBJ sono mm per default (`--mesh-units m` per una mesh già in metri).

Per un allineamento esatto fornire `--mesh-poses poses.json`, contenente una matrice
4×4 mesh-to-world in metri per ciascun episodio, ad esempio
`{"demo_0": [[...], [...], [...], [...]], "demo_1": ...}`.
In alternativa `{"mesh_to_world": [[...], [...], [...], [...]]}` applica una
trasformazione comune. Negli episodi senza trasformazione viene mostrata la posa nominale.
La matrice da registrare in Isaac è quella del sensore `mesh_to_organ_transform`
(`anatomy_processing.frames.read_isaac_mesh_to_world`), che include la calibrazione
acustica; la sola posa root dell'organo non è sufficiente.

## Organizzazione del codice

Il codice in `src/us_dp/` è diviso per responsabilità:

| Cartella / file | Contenuto |
| --- | --- |
| `dataset_generation/` | `collection.py`: recorder e adattatori Isaac; `oracle.py`: sweep e reach; `reach_demo.py`: dimostrazioni cinematiche; `synthetic.py`: dati sintetici e smoke test |
| `dataset/` | `processing.py`: formato episodi, validazione, split, fitting e DataLoader; `convert.py`: import HDF5 |
| `training/` | `model.py`: policy diffusion; `train.py`: training, checkpoint e valutazione |
| `deployment/` | `inference.py`: inferenza e replanning receding horizon |
| `anatomy_processing/` | `assets.py`: estrazione mesh; `frames.py`: frame e trasformazioni anatomiche; `viewer.py`: preparazione e visualizzazione Open3D |
| `common/` | `geometry.py`: geometria condivisa; `spline.py`: codec spline; `upstream.py`: integrazione con il checkout `spline_policy` |
| `config.py`, `cli.py`, `__main__.py` | Configurazioni e comandi pubblici |
| `_compat/` | Alias per i vecchi import Python; nessuna implementazione duplicata |

I comandi `us-dp ...` e `python -m us_dp ...` restano invariati.
Per nuovo codice usare, ad esempio, `from us_dp.training.train import train`
e `from us_dp.dataset_generation.collection import EpisodeRecorder`.
I vecchi import, ad esempio `from us_dp.train import train`, rimangono disponibili:
`us_dp.__path__` include `_compat`, i cui moduli rinviano allo stesso modulo canonico
senza caricare preventivamente Torch, Open3D o Isaac.
Configurazioni, dataset e run conservano i percorsi e i formati precedenti.


## Pipeline attuale: posizione TCP con Diffusion Spline Policy

```
3 immagini US + 3 pose cartesiane da 9 valori
  -> USFM ViT-B/16 congelato + MLP stato
  -> conditional U-Net 1D + DDPM sui parametri spline
  -> 15 coefficienti liberi XYZ
  -> decoder quadratico C1, origine locale vincolata a zero
  -> traiettoria cartesiana continua di 2 s
  -> conversione nel mondo + orientamento R0 fisso
  -> controller: esegue 0.4 s, poi nuova storia e nuovo piano
```

Lo stato preparato contiene posizione TCP(3) e orientamento TCP 6D(6):
`px, py, pz, r00, r10, r20, r01, r11, r21` (prime due colonne della rotazione).
`prepare` esclude q e dq se presenti nei raw; i prepared e checkpoint
precedenti a 23 valori restano caricabili. Per importare HDF5 con sola posa,
omettere `joint_position`, `joint_velocity` e `arm_joint_indices` dal mapping.
L'action representation contiene solo il moto futuro XYZ: né giunti né
orientamento sono predetti dalla spline.

Le dimostrazioni sono traiettorie TCP **misurate**; l'esperto può usare la
geometria privilegiata, la policy no. Si divide per configurazione phantom/anatomia
prima di creare finestre temporali. Normalizzazione solo sul training set.
Il fitting lavora nel frame del TCP corrente, lo stesso usato dal decoder
per tornare nel mondo in inferenza.

La loss di training è noise-prediction MSE DDPM sui cinque vettori liberi.
Validation e test separano diffusion loss, errore parametri, errore traiettoria
predetta e fitting error. Non è attiva una loss ausiliaria di traiettoria.

### Comandi

Dalla directory `us_dp`, impostare nel JSON usato da `prepare` il percorso
`usfm_pretrained` al checkpoint USFM e lasciare `usfm_freeze: true`.
Il training su dimostrazioni richiede questi pesi; i test sintetici possono
inizializzare il backbone senza checkpoint.

```bash
.venv/bin/us-dp prepare --raw data/raw --output data/prepared --config configs/default.json
.venv/bin/us-dp train --dataset data/prepared --output runs/train_001 --device cuda
.venv/bin/us-dp evaluate --checkpoint runs/train_001/best.pt --dataset data/prepared --device cuda
.venv/bin/python -m us_dp.training.dry_run --pretrained ../USFM/USFM_latest.pth
```

USFM deve essere importabile dal checkout adiacente (vedi `pyproject.toml`).
Il checkpoint policy include anche i pesi USFM e non richiede il file pretrained
originale per l'inferenza.

### Decoder e controller

```python
# free_params: [B,15], valori fisici nel frame TCP corrente
xyz_local = policy.codec(free_params, query_times_seconds, horizon_seconds=2.0)
# xyz_local: [B,T,3], frequenza di query indipendente da quella del dataset
```

`RecedingHorizonPolicy.observe` raccoglie la storia a 10 Hz;
`plan(control_hz=50)` restituisce 20 riferimenti di posizione assoluta nel mondo,
i relativi tempi da 0.02 a 0.4 s e `orientations_world`, fissati alla prima posa
del rollout. Chiamare `reset()` al cambio episodio. Ogni piano riparte dal TCP
misurato. L'orientamento fisso non viene appreso o interpolato dalla policy.

Il server i4h negozia lunghezza/cadenza della storia con il proxy remoto, che
raccoglie osservazioni reali anche durante l'esecuzione. Il messaggio trasporta
q e dq; non duplica un frame per simulare la storia.

Audit con file/classi, parametrizzazione, metriche, shape e verifica:
[docs/pipeline_audit.md](docs/pipeline_audit.md).
Report di un batch eseguito: [docs/dry_run_example.json](docs/dry_run_example.json).
Dettagli di raccolta e anatomia: [impl_details.md](impl_details.md).

Il dry-run verifica il software. La qualità della scansione richiede ancora
rollout Isaac e misure di copertura, tracking e contatto.

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
