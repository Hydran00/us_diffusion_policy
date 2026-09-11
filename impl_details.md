# Ultrasound Diffusion Spline Policy

Prima implementazione della pipeline descritta in
[README_ultrasound_diffusion_spline_policy.md](../README_ultrasound_diffusion_spline_policy.md).
Il codice applicativo è in `us_dp`; la repo locale `../spline_policy` è una dipendenza importata direttamente.

La pipeline implementata è:

```text
ROI di contatto sopra l'organo + posa phantom (privilegiati)
  -> sweep oracolo -> controller del simulatore
  -> B-mode + stato robot + pose TCP effettivamente eseguite
  -> split per configurazione phantom/anatomia
  -> finestre future nel frame TCP corrente
  -> fitting spline C1 con punto iniziale vincolato
  -> DDPM condizionata da storia ultrasound + propriocezione
  -> spline locale -> riferimenti cartesiani nel mondo con timestamp
  -> esecuzione del prefisso e nuova osservazione
```

Sono eseguibili preparazione dei dati, training, valutazione offline e inferenza.
Sono disponibili un generatore di sweep da ROI superficiale, un recorder e un
adattatore per leggere la scena i4h `panda_phantom`. La raccolta richiede di
collegare il controller Isaac attraverso le callback descritte sotto: questo
package **non crea né avvia una nuova scena Isaac**. Non è ancora stata effettuata
una validazione di scansione a circuito chiuso nel simulatore.

## Installazione

Dalla directory `us_dp`, in un ambiente Python separato dallo stack Isaac:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,hdf5]'
export SPLINE_POLICY_ROOT="$(realpath ../spline_policy)"
```

La versione `diffusers==0.30.0` segue l'ambiente dichiarato dalla repo upstream.
Vengono importati soltanto `QuadraticSpline` e `ConditionalUnet1D`: non sono
necessari Robomimic, Push-T, Gym o l'intero ambiente dei benchmark upstream.
`SPLINE_POLICY_ROOT` punta alla radice che contiene il README della repo;
in alternativa usare `us-dp --spline-policy-root /percorso/spline_policy ...`.
Il default cerca la repo sorella del checkout `us_dp`.

## Prima verifica su CPU

```bash
us-dp smoke --output runs/smoke_001
python -m pytest -q
```

`smoke` produce immagini **sintetiche di test**, sei episodi, dataset diviso per
configurazione, una singola epoca di training, `best.pt`, valutazione su test e
`prediction.npz`. Il report è `runs/smoke_001/report.json`.
Le immagini non provengono dal simulatore acustico; loss ed errori di questa
prova non misurano la capacità di scansione. Usare una directory nuova per ogni
esperimento: i comandi non sovrascrivono dataset o run precedenti.

## Dati reali dal simulatore

Ogni file `.npz` rappresenta un episodio sincronizzato:

| Campo | Forma / significato |
|---|---|
| `timestamps` | `(T,)`, secondi di simulazione, monotoni, passo costante |
| `ultrasound` | `(T,H,W)`, grayscale `uint8`, finestra fissa −60…0 dB |
| `robot_state` | `(T,D)`, solo osservazioni disponibili al robot |
| `probe_pose` | `(T,4,4)`, trasformazione TCP → mondo, metri |
| `metadata` | JSON con `episode_id`, `group_id`, `state_fields`, `source` |

Lo stato predefinito ha 23 componenti: 7 posizioni articolari, 7 velocità,
3 coordinate TCP e le prime due colonne della rotazione TCP → mondo.
L'ordine delle colonne è `STATE_FIELDS` in `collection.py`. Forza normale e
contatto possono essere aggiunti esplicitamente al contratto dati; non vengono
inventati quando il simulatore non li fornisce.

`group_id` deve essere uguale per rollout della stessa configurazione
phantom/anatomia, anche con direzioni di scansione diverse. Servono almeno tre
gruppi indipendenti. Lo split avviene **prima** della costruzione delle finestre;
normalizzazione di stati e parametri usa soltanto il training set. Le frazioni
sono approssimate a gruppi interi, con almeno un gruppo validation e uno test.
Metadati geometrici privilegiati non sono forniti alla policy. Non passare
`task_obs` indiscriminatamente: può contenere la posa del target.

### Lettura della scena i4h esistente

Nel checkout attuale il B-mode reale è disponibile in `panda_phantom` con
ultrasound abilitato; `ultrasound_probe_reach` è un task di raggiungimento.
`read_isaac_observation` usa `ultrasound.data.output['bmode_db']`, le articolazioni
Franka e `ee_frame` come frame TCP del controller. La trasformazione del sensore
acustico rimane gestita dalla scena. Non scambiare TCP e frame acustico.

Nel processo Isaac, rendere importabile `us_dp/src` senza installare lo stack
policy nell'ambiente arena. Dopo l'inizializzazione e il binding del sensore:

```python
from us_dp.dataset_generation.collection import EpisodeRecorder, read_isaac_observation

recorder = EpisodeRecorder(
    "data/raw/episode_0000.npz",
    episode_id="episode_0000",
    group_id="phantom_configuration_0000",
    phantom_pose=phantom_pose.tolist(),  # solo metadata
)

# Dentro il loop di simulazione, dopo lo step e su un nuovo frame ultrasound:
observation, frame_id = read_isaac_observation(
    env,
    timestamp=simulation_time,
    quaternion_order="xyzw",  # convenzione del checkout i4h corrente
)
recorder.append(**observation)
# A fine episodio riuscito:
recorder.save()
```

Registrare a 10 Hz per la configurazione predefinita, una sola volta per nuovo
`frame_id`, dopo lo step del simulatore. Il chiamante deve verificare che B-mode,
stato e TCP corrispondano allo stesso istante; non rinominare un frame vecchio
con un timestamp nuovo. Su versioni Isaac con quaternioni `wxyz`, specificarlo.
Una discontinuità temporale o un reset deve terminare l'episodio.

### Generazione dello sweep esperto

`oracle.surface_sweep(points, normals, phantom_pose, samples, offset_m)` genera
uno sweep quadratico con avvio/arresto graduali lungo l'asse principale della ROI.
`random_phantom_pose` randomizza traslazione planare e yaw rispetto alla posa
nominale. Applicare la medesima trasformazione al phantom reale in scena.

La ROI deve essere una fascia della **superficie esterna del phantom sopra
l'organo**, espressa in metri nel frame phantom, con normali esterne. La mesh
interna dell'organo serve a scegliere la regione da scansionare, non è una
superficie su cui posizionare direttamente la sonda. La selezione/proiezione
della ROI dalla geometria della scena resta specifica dell'asset.

Il generatore approssima la superficie con un fit quadratico: verificare
geometricamente lo sweep sulla propria ROI prima di eseguirlo. Le normali sono
riferimenti per il controller calibrato, non quaternioni TCP già pronti.

```python
from us_dp.dataset_generation.collection import collect_oracle_episode
from us_dp.dataset_generation.oracle import surface_sweep

reference = surface_sweep(skin_roi_points, skin_roi_normals, phantom_pose, samples=101)
# Portare prima il TCP all'inizio dello sweep con il controller della scena.
collect_oracle_episode(
    recorder,
    reference,
    observe=read_current_synchronized_observation,
    execute_reference=execute_cartesian_reference,
    sample_hz=10.0,
)
```

Le due callback sono fornite dal loop del simulatore:

- `observe()` restituisce il dizionario `observation` mostrato sopra;
- `execute_reference(position_world, normal_world, dt)` esegue IK/regolazione
  del contatto per esattamente `dt` secondi simulati e segnala terminazione,
  reset o fallimento del controllo tramite eccezione.

Il recorder salva le **pose misurate**, inclusi gli errori di tracking.
Le callback vanno collegate al driver/controller della scena; i Task i4h leggono
`ctx.scene`, scrivono `ctx.act` e non chiamano direttamente `env.step()`.

### Import da HDF5 i4h

```bash
us-dp import-hdf5 \
  --input /percorso/recording.hdf5 \
  --mapping configs/hdf5_mapping.example.json \
  --output data/raw_imported
```

Il mapping è un esempio da adattare ai dataset realmente registrati. Sono
necessarie pose TCP misurate, immagini, propriocezione e `group_id` per demo.
Le registrazioni standard potrebbero non contenere velocità o pose TCP: il
convertitore segnala i campi mancanti. Non usa `actions` come sostituto della
traiettoria eseguita. Indicare esplicitamente indici dei sette giunti, formato
immagine e ordine quaternion. Usare `sample_hz` al posto di `timestamps` nel
mapping solo quando la cadenza di registrazione è nota e costante.

## Preparazione, training e valutazione

```bash
us-dp prepare --raw data/raw --output data/prepared --config configs/default.json
us-dp train --dataset data/prepared --output runs/train_001 --device cuda
us-dp evaluate --checkpoint runs/train_001/best.pt --dataset data/prepared --device cuda
```

Per una prova breve usare `--epochs 1 --batch-size 8` e `--device cpu`.
Il config del dataset viene salvato nel manifest e riutilizzato dal training.
Il checkpoint contiene pesi, statistiche, configurazione, ordine dello stato,
ottimizzatore e commit della repo spline. `best.pt` è selezionato tramite loss
sul validation set; `last.pt` rappresenta l'ultima epoca. Il test rimane separato.
Non è ancora disponibile un comando di ripresa del training.

La valutazione riporta errore cartesiano rispetto alle traiettorie esperte
registrate e residuo del fitting spline. Successo, copertura dell'organo e
qualità del contatto richiedono rollout reali nel simulatore.

## Inferenza con ripianificazione

```python
from us_dp.deployment.inference import RecedingHorizonPolicy

runner = RecedingHorizonPolicy("runs/train_001/best.pt", device="cuda")
# Per ogni campione sincronizzato a 10 Hz, anche durante l'esecuzione:
runner.observe(image_uint8, robot_state, probe_pose_world, simulation_time)
# Dopo almeno tre osservazioni, a ogni ripianificazione:
plan = runner.plan(control_hz=50)
# Il controller esegue i 20 riferimenti per i prossimi 0.4 s:
positions = plan["positions_world"]
times = plan["time_from_start"]
```

Il decoder restituisce riferimenti di **posizione TCP assoluta nel mondo**, non
azioni articolari né delta cartesiani direttamente accettabili da un task.
Il controller deve convertirli nel proprio contratto d'azione, gestire
orientamento/contact e rispettare i tempi. A ogni reset chiamare `runner.reset()`.
Un gap nelle osservazioni richiede di ricostruire la storia, evitando di
alimentare frame distanziati come se fossero consecutivi.

La spline ha continuità C1 fra segmenti e parte dal TCP misurato a ogni
ripianificazione. La continuità della **velocità fra due piani successivi** non
è ancora vincolata. Inoltre, la API sincrona non compensa la latenza di
inferenza: durante una distribuzione online asincrona il driver deve gestire
l'età dell'osservazione e del piano. Mantenere policy e Isaac in processi
separati; l'integrazione i4h remota va esposta tramite `i4h_common.server.PolicyServer`
quando sarà collegato il controller, non importando la policy nella scena.

## Riutilizzo di Spline Policy e scelte iniziali

| Componente | Implementazione |
|---|---|
| Base quadratica Bernstein e vincoli C1 | `QuadraticSpline` upstream, matrici `C` e `phi` |
| Denoiser condizionato | `ConditionalUnet1D` upstream |
| Fitting locale con ancora e decoder differenziabile | `us_dp.common.spline`, usando le matrici upstream |
| DDPM epsilon e campionamento | `us_dp.training.model` + scheduler Diffusers |
| Encoder B-mode e stato | CNN grayscale con storia impilata + MLP |
| Dataset, normalizzazione, checkpoint, rollout API | `us_dp` |

Con quattro segmenti la repo upstream usa sei parametri indipendenti per asse.
Il primo è fissato a zero nel frame locale: la diffusione predice i restanti
cinque vettori 3D. La sequenza viene padded internamente per le dimensioni della
U-Net, senza includere il padding nella loss o nel decoder.

La policy originale upstream usa una loss sulla traiettoria decodificata;
questa implementazione usa la predizione del rumore su parametri ottenuti dal
fitting, come richiesto dal README di progetto. Il CNN compatto serve da primo
encoder; ResNet-18, baseline BC, loss ausiliaria sulla traiettoria e
randomizzazione anatomica/acustica avanzata sono estensioni successive.
Il codice upstream resta nel suo checkout con le proprie licenze.

## Verifica eseguita

Verificati su CPU: 10 test passati, lint e formattazione, build della wheel e
smoke completo con salvataggio/ricaricamento del checkpoint. Il report della
prima esecuzione locale è in `runs/initial_smoke/report.json` (artefatto ignorato
da Git). Non sono stati eseguiti rollout Isaac né training su episodi reali.

## Localizzazione del fegato e visualizzatore Open3D

Il primo blocco geometrico automatico usa **Skin.obj e Liver.obj del container
ultrasound reale**. Calcola centro e assi principali sulla superficie completa
del fegato; la semplificazione riguarda soltanto le mesh mostrate a schermo.

Nella `.venv` del progetto:

```bash
python -m pip install -e '.[viewer]'
# Solo se non sono già stati estratti gli asset:
us-dp extract-anatomy --output data/assets/abdphantom
# Calcolo dei landmark e apertura della finestra (directory di output nuova):
us-dp anatomy --mesh-dir data/assets/abdphantom --output runs/anatomy_002
```

In questa workspace gli asset sono già in `data/assets/abdphantom`, i landmark
sono già calcolati in `runs/anatomy_001`. Per riaprire subito il risultato:

```bash
us-dp view-anatomy --input runs/anatomy_001
```

La finestra mostra:

- phantom esterno in wireframe grigio;
- fegato in wireframe arancione;
- centro stimato come sfera gialla;
- i **primi due assi principali del phantom intero (Skin)**: 1 rosso, 2 verde,
  in ordine di varianza decrescente, traslati al centro del fegato come target.
  Gli assi PCA del fegato non sono visualizzati.

Ruotare con il mouse e usare la rotella per lo zoom. **P** alterna phantom
wireframe/solido/nascosto; **L** alterna fegato wireframe/solido/nascosto;
**S** salva `viewer.png`; **Q** chiude. Il wireframe iniziale permette di vedere
il centro dentro il fegato. Gli assi sono geometrici, non etichette anatomiche:
la PCA non identifica da sola direzioni cliniche destra/sinistra o craniale/caudale.

La finestra usa il [visualizzatore Open3D con callback da tastiera](https://www.open3d.org/docs/release/python_api/open3d.visualization.VisualizerWithKeyCallback.html).
Serve una sessione grafica; da un terminale senza display si può eseguire soltanto
il calcolo con `us-dp anatomy ... --no-view`. Non avviare questa finestra dal
thread di controllo del robot: è un processo separato.

### Definizione del centro e degli assi

`anatomy.estimate_liver_frame(vertices, triangles)` integra esattamente i primi
due momenti di ogni triangolo, pesandoli per area. Il centro è quindi il
**baricentro della superficie**, non il centro di massa volumetrico. Questo
metodo evita il bias del conteggio dei vertici nelle zone più triangolate e
funziona anche con mesh non chiuse. Le unità di ingresso dell'API sono metri.

Il loader converte gli OBJ originali da millimetri a metri (`--mesh-units mm`).
I segni degli assi vengono fissati nel frame originale della mesh e la terna
è destrorsa. A ogni episodio si trasforma la terna già calcolata, senza rifare
la PCA nel mondo: ciò evita cambi di segno artificiali quando ruota il phantom.
Il report segnala eventuali autovalori quasi uguali (`ambiguous_axes`).

I file `runs/anatomy_001/landmarks.json`, `Skin.ply`, `Liver.ply` contengono
landmark, unità, hash degli OBJ, posa applicata e mesh leggere per il viewer.
Le coordinate **senza `--pose` sono nel frame della mesh acustica**, non nelle
coordinate del robot. Il centro misurato sugli asset attuali è circa
`[-0.049697, -0.024022, 0.049813]` metri in quel frame.

### Collegamento al frame del simulatore

Dopo un reset e l'aggiornamento dei sensori nella scena `panda_phantom`:

```python
import json
from pathlib import Path
from us_dp.anatomy_processing.frames import LiverFrame, read_isaac_mesh_to_world

report = json.loads(Path("/percorso/us_dp/runs/anatomy_001/landmarks.json").read_text())
liver_local = LiverFrame.from_dict(report["liver_mesh_frame"])

# Ripetere dopo ogni randomizzazione del phantom:
mesh_to_world = read_isaac_mesh_to_world(env, quaternion_order="xyzw")
liver_world = liver_local.in_world(mesh_to_world)
center_world = liver_world.center
principal_axes_world = liver_world.axes  # assi nelle COLONNE

# Esportazione facoltativa della posa per il viewer esterno:
Path("/percorso/mesh_pose.json").write_text(
    json.dumps({"mesh_to_world": mesh_to_world.tolist()})
)
```

L'adattatore legge **`mesh_to_organ_transform`**, esattamente il riferimento
usato dal renderer B-mode i4h; leggere solo `organs.root_pose` perderebbe l'offset
di calibrazione della mesh. La convenzione quaternion è esplicita e va adattata
se cambia la versione Isaac. Skin e Liver devono provenire dallo stesso set di
asset effettivamente caricato dal renderer.

Per visualizzare la posa esportata:

```bash
us-dp anatomy --mesh-dir data/assets/abdphantom \
  --pose /percorso/mesh_pose.json --output runs/anatomy_world_001
```

Questo blocco fornisce il target anatomico privilegiato per l'esperto. Non
sposta ancora phantom o robot: restano da implementare la proiezione del target
sulla superficie esterna, il posizionamento iniziale con verifica del contatto
e l'esecuzione dello sweep. Il centro interno del fegato **non** è un comando
di posizione TCP.

### Orientamenti di riferimento del phantom

Il viewer usa ora la PCA della superficie **Skin completa**, calcolata con la
stessa integrazione pesata per area: il primo e il secondo asse sono i
riferimenti geometrici per longitudinale e trasversale. Le frecce sono ancorate
al centro del fegato per mostrare le direzioni al target anatomico; non indicano
il centro del phantom. Il terzo asse resta nel report per completare la terna,
ma non viene visualizzato. Queste direzioni non definiscono ancora una posa
TCP completa: serviranno anche la normale locale e la calibrazione della sonda.

`landmarks.json` contiene ora `phantom_mesh_frame` e `phantom_display_frame`,
oltre ai campi del fegato. I vecchi report vengono aggiornati alla prima apertura
leggendo la Skin originale e verificandone l'hash. La PCA non viene stimata dalla
mesh semplificata del viewer.

Nel runtime, dopo ogni reset:

```python
phantom_local = LiverFrame.from_dict(report["phantom_mesh_frame"])
phantom_world = phantom_local.in_world(mesh_to_world)
longitudinal_direction = phantom_world.axes[:, 0]
transverse_direction = phantom_world.axes[:, 1]
# Il target anatomico rimane liver_local.in_world(mesh_to_world).center.
```

Il nome `LiverFrame` è mantenuto per compatibilità e rappresenta una terna PCA
generica; `estimate_surface_frame` può essere usato su entrambe le mesh.
