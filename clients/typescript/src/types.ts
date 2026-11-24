// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: MIT

// The HTTP API's request and reply types, generated from
// api/openapi.yaml by api/generate.py. Do not edit.

export const VERSION = "gf/v1";

/** a resource name */
export type Name = string;

/**
 * a catalog model's name, its id without `general-forecast/`, such as
 * `tabpfn-3.5-classifier`; unlike a chosen resource name it may start with a digit and
 * hold dots
 */
export type ModelName = string;

/**
 * the model a run publishes, `<project>/<name>@ft<version>`, which names the run too:
 * the fine-tune it trains and the version it publishes, as the catalog and a
 * predictor's `model` name that version
 */
export type RunModel = string;

/**
 * a run by its `model` without the project, which the path names, `credit-default@ft2`,
 * or a fine-tune's name alone, `credit-default`, for the name's latest run
 */
export type RunRef = string;

/**
 * a project's fine-tune version by its name and version, as `credit-default@ft3` names
 * `<project>/credit-default@ft3`
 */
export type FineTuneName = string;

/**
 * a catalog model id, `general-forecast/<name>`, or a derived model's,
 * `<owner>/<name>@ft<version>`; then an optional digest pin (`@sha256:<digest>`) and
 * variant suffix (`@4*2`)
 */
export type ModelId = string;

export type Digest = string;

export type Task = "classification" | "regression" | "forecast";

export type ResourceKind = "predictor" | "model" | "project" | "api-key" | "secret" | "context" | "source" | "workspace" | "snapshot" | "run" | "prediction" | "experiment" | "estimate";

/** what the envelope is about; GET lists use the collection name */
export interface Resource {
  kind: ResourceKind;
  name: string;
}

/** the fields every request and reply carries */
export interface Envelope {
  api_version: "gf/v1";
  project: Name;
  request_id: string;
  resource: Resource;
}

/**
 * ValidationError 422, NotFound 404, Unauthorized 401, CapacityExceeded 422,
 * ModelNotReady 503 (a model, or the runtime or pool serving it, is still loading,
 * starting or compiling), Internal 500, Conflict 409 (a compare-and-swap on a stale
 * generation), Unsupported 501 (an operation this server does not offer, such as a
 * hosted route on gf serve), PaymentRequired 402 (the workspace's plan or credit does
 * not cover the request), AmbiguousReference 422 (a short model name whose task nothing
 * decides: pass `task`, or the full id), Unavailable 503 or 504 (a service the request
 * needs, such as a database, storage or a control plane, is down for now or did not
 * answer in time). Every 503 carries `Retry-After`, and a client may retry it, as it
 * may a 504. A code the server raises that is not in this enum is sent as the enum code
 * it folds into, with the code itself, in snake case, as `reason`:
 * `CredentialUnavailable` 503 is `Unavailable` with reason `credential_unavailable`.
 * Hosted fine-tuning's gates, with their `reason`: `PaymentRequired` 402
 * `plan_required` (fine-tuning, and the resumable uploads and synthetic batches that
 * feed it, need a Pro or Enterprise plan) and `credits_exhausted` (an enforced
 * workspace has used its credits, which also refuses predictions); `Unauthorized` 403
 * `fine_tuning_unavailable` (fine-tuning is not turned on for the project);
 * `CapacityExceeded` 429 `run_limit_reached` (the project has as many runs queued or
 * running as it may), `quota_exceeded` (the project's training data would pass its
 * quota) and `snapshot_quota_exceeded`. Those three 429s carry `limit` and `actual`,
 * and lift only when the caller ends a run or deletes data, so clients do not retry
 * them; a 429 `rate_limited` passes on its own and may be retried. A run's create is
 * refused with `Conflict` 409 `run_active` (another run of the fine-tune is still
 * queued or running) or `base_mismatch` (the fine-tune's versions train from another
 * base), and with `ValidationError` 422 `base_is_fine_tune` (a fine-tune sent as the
 * base).
 */
export const ERROR_CODES = ["ValidationError", "NotFound", "Unauthorized", "CapacityExceeded", "ModelNotReady", "Internal", "Conflict", "Unsupported", "PaymentRequired", "AmbiguousReference", "Unavailable"] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

/**
 * `param` names the payload field (`target`, `inputs.plan`, `queries`); every limit
 * error carries `limit` and `actual` and names the model in `message`. The request id
 * is in the envelope.
 */
export interface ApiError {
  code: ErrorCode;
  message: string;
  reason?: string;
  param?: string;
  hint?: string;
  limit?: number;
  actual?: number;
}

export interface ErrorReply extends Envelope {
  error: ApiError;
}

/** one record; column names are keys, cells are scalars */
export type Row = Record<string, string | number | boolean | null>;

/**
 * a column's kind, the engine's vocabulary an ad hoc call's `inputs.types` takes: whole
 * numbers, real numbers, 0/1 flags, categories, free text, dates and times, and
 * booleans
 */
export type Kind = "int" | "float" | "0/1" | "category" | "text" | "timestamp" | "bool";

/**
 * inputs by rule rather than by list: `exclude` removes columns from the inferred set,
 * `types` overrides inferred types; a forecaster's `past` and `future` are inputs
 * whatever else is, and its other inputs are past covariates
 */
export interface Inputs {
  exclude?: string[];
  types?: Record<string, Kind>;
  past?: string[];
  future?: string[];
}

export type ColumnType = "number" | "category" | "text" | "boolean" | "date" | "timestamp";

/**
 * which context rows a query sees; fractions below 1 are of the model's row cap,
 * integers are rows; `core: null` is the whole table
 */
export interface Selection {
  core?: number | null;
  neighbors?: number;
  kind?: "stratified" | "random";
  recent?: number | null;
}

/**
 * a model by its short name, such as `tabicl-v2`, or its full id; a short name resolves
 * to its family's entry for the task given, else the task inferred from the target
 * column, else the family's only task
 */
export type ModelField = ModelId | ModelName;

/** a context the service holds by its digest */
export interface SnapshotReference {
  snapshot: Digest;
}

/**
 * the ad hoc predictor: a predictor's fields, with `context` as rows or by reference to
 * a snapshot the service holds (`{"snapshot": "<digest>"}`, uploaded with `PUT
 * /snapshots?digest=`), and `queries` as rows, or as the Arrow stream the request is.
 * `exclude`, `types`, `past` and `future` take a predictor spec's forms beside
 * `inputs`, or sit in `inputs` as `Inputs`, once. The target column decides a short
 * model name's entry; there is no `task`
 */
export interface Manifest {
  exclude?: string[];
  types?: Record<string, ColumnType>;
  past?: string[];
  future?: string[];
  model: ModelField;
  target: string;
  id?: string;
  inputs?: string[] | Inputs;
  time?: string;
  series?: string[];
  related?: "all" | string[];
  frequency?: string | number;
  horizon?: number;
  quantiles?: number[];
  outputs?: Array<"label" | "value" | "probabilities" | "quantiles">;
  selection?: Selection;
  context: Row[] | SnapshotReference;
  queries: Row[];
}

/**
 * what a call is billed: its scored rows, and past the large-context line
 * (`large_context_cells` in the price book) the GPU seconds of its own execution.
 * `calls` is the model calls the engine made to answer it, its result's `timing.calls`
 */
export interface Usage {
  calls?: number;
  rows: number;
  credits: number;
  usd: number;
  gpu_seconds?: number | null;
}

/**
 * where a call's time went, queueing and cold start apart from compute;
 * `compile_seconds` is the time its steps spent compiling or loading programs they
 * waited on, which `compute_seconds`, its own execution, leaves out
 */
export interface Timing {
  calls: number;
  queued_seconds: number;
  cold_start_seconds: number;
  compile_seconds?: number;
  compute_seconds: number;
}

export interface HeldSnapshot {
  digest: Digest;
  rows: number;
}

export interface HeldSnapshotReply extends Envelope {
  result: HeldSnapshot;
}

export interface ManifestRequest extends Envelope {
  payload: Manifest;
}

/**
 * for a live context, score against the rows accepted at or before this instant: an RFC
 * 3339 time (one with no offset is UTC), or a number of seconds since the epoch.
 * Anything else is a `ValidationError` whose `param` is `as_of`. A snapshot or request
 * context refuses it the same way.
 */
export type AsOfRequest = string | number;

/**
 * a predictor's payload: the queries, `as_of` for a live context, and `context` for a
 * predictor whose lock takes its context with each request, which only gf serve serves.
 * The environment is the route's `?env=`.
 */
export interface Queries {
  queries: Row[];
  as_of?: AsOfRequest;
  context?: Row[];
}

export interface QueriesRequest extends Envelope {
  payload: Queries;
}

export type ContextKind = "snapshot" | "request" | "live";

/**
 * an instant as every reply carries it: an RFC 3339 UTC string such as
 * `2026-10-27T08:00:00Z`, or null
 */
export type AsOf = string | null;

/**
 * the context a prediction actually used, the same shape from gf serve and the hosted
 * platform. `source` is the context's kind; `name` names a live context; `version` is a
 * live context's accepted sequence; `digest` is a snapshot's content digest; `rows`
 * counts the rows read; `as_of` is the instant a live context was read as of. Each
 * field a source does not have is null.
 */
export interface ContextInfo {
  source: ContextKind;
  name: Name | null;
  version: number | null;
  digest: Digest | null;
  rows: number;
  as_of: AsOf;
  used?: "exact" | "retrieved";
  core?: number | null;
  cells?: number;
}

/**
 * the query's id under the predictor's id column name when it names one (omitted when
 * it does not); `label` and `probabilities` for classification, `value` for regression,
 * `value` and `quantiles` for forecasts
 */
export interface Prediction {
  label?: string | number | boolean;
  value?: number;
  probabilities?: Record<string, number>;
  quantiles?: Record<string, number>;
  [key: string]: string | number | boolean | Record<string, number> | undefined;
}

export interface Result {
  model: ModelId;
  task: Task;
  context: ContextInfo;
  predictor?: Name | null;
  revision?: number | null;
  environment?: Name | null;
  predictions: Prediction[];
  usage?: Usage;
  timing?: Timing;
  notes?: string[];
}

export interface ResultReply extends Envelope {
  result: Result;
}

/** a prediction the hosted plane accepted, `pred-` and 24 hex digits */
export type PredictionId = string;

/**
 * a prediction the hosted plane accepted and is still working on: `starting` while
 * hosted inference starts, `running` once it has taken the work
 */
export interface PredictionStatus {
  id: PredictionId;
  state: "starting" | "running";
  created: number;
  retry_after: number;
}

export interface PredictionAccepted extends Envelope {
  result: PredictionStatus;
}

/**
 * the context sizes a model was trained and measured to, its own as its catalog entry
 * gives them: a call past them is admitted by the device's memory and carries a note.
 * No count of queries is a limit: a call's queries are decoded in as many model calls
 * as the device needs, which `timing.calls` counts
 */
export interface Limits {
  context_rows: number;
  inputs: number;
}

/** a compiled program's shape */
export interface Shape {
  context_rows: number;
  queries: number;
  inputs: number;
  classes?: number;
}

/**
 * what a catalog model saw in training: its upstream's stated context rows, inputs and
 * classes, each null where none is stated. A note shown beside results, never a limit;
 * null for a forecaster
 */
export interface TrainedOn {
  rows: number | null;
  inputs: number | null;
  classes: number | null;
}

/**
 * what the weights change of the base: low-rank adapters on its layers (`lora`), an
 * input adapter in front of it (`input_adapter`), or the named components' own weights
 * (`full`), the names a run's `changes` use. A version trained with LoRA beside an
 * input adapter is `input_adapter`: the LoRA is stored with it.
 */
export type FineTuneKind = "lora" | "input_adapter" | "full";

/**
 * the training step whose weights the version keeps: the one that scored best on the
 * rows the run held out. Step 0 is the base itself: no check during training beat the
 * base, so the version predicts exactly as its base does. Null for a version whose run
 * did not say.
 */
export type KeptStep = number | null;

/**
 * a standard component kind of the shared attention block, or an addition to a model's
 * cell embeddings, the same in every model
 */
export type RunArchitectureComponent = "attention.kv_heads" | "attention.query_scale" | "attention.qk_norm" | "attention.output_gate" | "positional_encoding" | "norm" | "norm_placement" | "feed_forward" | "attention.temperature" | "feed_forward_branch" | "cell_embedding.numeric" | "cell_embedding.categorical";

/**
 * What a component becomes: `{"type": <option>}` with the option's settings beside it,
 * as `{"type": "grouped", "heads": 2}` or `{"type": "rope", "theta": 10000}`. A catalog
 * model's `architecture` lists each component's options, their settings and defaults; a
 * setting left out takes its default, and an unknown one is refused with 422.
 */
export interface RunArchitectureOption {
  type: string;
  [key: string]: number | number | boolean | null | string;
}

/** an init type, as a catalog model's `architecture.inits` lists them */
export type RunArchitectureInitType = "default" | "zeros" | "ones" | "constant" | "normal" | "truncated_normal" | "uniform" | "xavier_uniform" | "xavier_normal" | "kaiming_uniform" | "kaiming_normal" | "orthogonal";

/**
 * How a change's new parameters start, each matching its `torch.nn.init` /
 * `jax.nn.initializers` namesake: `default` (the option's documented init), `zeros`,
 * `ones`, `constant` (`value`), `normal` (`mean` 0 and `std` 0.02 when left out),
 * `truncated_normal` (`std` 0.02, cut at 2 standard deviations), `uniform` (`low` 0 and
 * `high` 1), and `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`
 * (fan in, ReLU gain) and `orthogonal`, which draw the change's matrices while its
 * vectors take the option's default. Draws are deterministic in the run's `seed`, the
 * component and the stage. Every init is accepted for every option with parameters.
 */
export interface RunArchitectureInit {
  type: RunArchitectureInitType;
  [key: string]: number | RunArchitectureInitType;
}

/**
 * A change as the run resolved it and its version records it: the stages it applies in,
 * the option with every setting filled in (a null `hidden` is the block's own width),
 * and its init with every setting filled in, null for an option without parameters
 */
export interface RunResolvedArchitectureChange {
  component: RunArchitectureComponent;
  stages: string[];
  to: RunArchitectureOption;
  init: RunArchitectureInit | null;
}

/**
 * what a fine-tune is made of, as a predictor's lock records it: its base, pinned, and
 * its weights; present on the project's fine-tunes alone, which otherwise describe
 * themselves as their base does
 */
export interface DerivedModel {
  base: ModelId;
  base_digest: Digest;
  digest: Digest;
  bags: number;
  kind: FineTuneKind;
  version: number;
  created: number;
  kept_step: KeptStep;
  architecture?: RunResolvedArchitectureChange[];
}

/**
 * one published version of a fine-tune: its id, its weights' digest and the step its
 * run kept. The run that trained it has the same id as its `model`. A version never
 * changes: its id names the same weights for as long as it exists.
 */
export interface FineTuneVersion {
  name: ModelId;
  digest: Digest;
  version: number;
  kind: FineTuneKind;
  created: number;
  kept_step: KeptStep;
}

export interface Artifact {
  repository: string;
  revision: string;
  digest: Digest;
  license: string;
  digest_format?: string;
  files: Record<string, string>;
  [key: string]: unknown;
}

/** a catalog variant's ensemble, `@K` codes and `*N` estimators */
export interface ModelVariant {
  codes: number;
  estimators: number;
}

export interface Capability {
  name: string;
  state: "supported" | "unsupported" | "unconfigured";
  code?: string | null;
  reason?: string | null;
}

/**
 * one setting of an option or an init; `default` is its value when left out (null for a
 * width the block gives), absent for a `required` one
 */
export interface ArchitectureSetting {
  name: string;
  type: "integer" | "number" | "boolean";
  description: string;
  default?: number | number | boolean | null;
  minimum?: number;
  required?: boolean;
}

/**
 * one option of a component, its settings, and how its new parameters start under the
 * `default` init; `default_init` is null for an option without parameters, which takes
 * no init
 */
export interface ArchitectureOption {
  type: string;
  description: string;
  settings: ArchitectureSetting[];
  default_init: string | null;
}

/**
 * options of a component the model cannot take in some stages, and why; empty `options`
 * is every option, empty `stages` every stage
 */
export interface ArchitectureUnavailable {
  options: string[];
  stages: string[];
  reason: string;
}

/** a component kind, its options and what this model cannot take of it */
export interface ArchitectureComponent {
  component: RunArchitectureComponent;
  description: string;
  options: ArchitectureOption[];
  unavailable: ArchitectureUnavailable[];
}

/**
 * an init type; a `fan_based` one draws matrices alone, a change's vectors then taking
 * the option's default
 */
export interface ArchitectureInit {
  type: string;
  description: string;
  settings: ArchitectureSetting[];
  fan_based: boolean;
}

/**
 * what a run's `architecture` may change of a catalog model: its stages, every standard
 * component kind with its options, their settings and default inits and what the model
 * cannot take, and the init types. Present on the catalog models a run may change, the
 * ones that can be fine-tuned.
 */
export interface ModelArchitecture {
  stages: string[];
  components: ArchitectureComponent[];
  inits: ArchitectureInit[];
}

/**
 * the catalog's description of a served model: `model` is the resolved id with its
 * content digest, `limits` the model's own sizes as the catalog gives them (the same
 * from gf serve and the hosted plane), and `warm_shapes` the shapes compiled at
 * startup, empty where none were
 */
export interface Model {
  short: string | null;
  kind: "catalog" | "fine-tune";
  family?: string;
  base: string | null;
  trained_on: TrainedOn | null;
  hostable?: boolean;
  model: string;
  name: ModelId;
  derived?: DerivedModel;
  versions?: FineTuneVersion[];
  task: Task;
  repository: string;
  revision: string;
  license: string;
  runtime: "cpu" | "gpu";
  artifact: Artifact;
  artifact_verification?: string;
  license_source?: string;
  variant?: ModelVariant;
  bounds: Record<string, number[] | number | number | null>;
  capabilities: Capability[];
  limits: Limits;
  warm_shapes: Shape[];
  architecture?: ModelArchitecture;
}

export interface ModelList {
  models: Model[];
}

export interface ModelReply extends Envelope {
  result: Model;
}

export interface ModelListReply extends Envelope {
  result: ModelList;
}

export interface Build {
  commit: string | null;
  version: string | null;
  gf_commit: string | null;
  gf_version: string | null;
}

export interface Health {
  status: "ok";
  build?: Build;
}

export interface HealthError {
  error: ApiError;
}

export interface Empty {
}

export interface EmptyRequest extends Envelope {
  payload: Empty;
}

export interface Project {
  name: Name;
  organization: Name;
}

export interface ProjectReply extends Envelope {
  result: Project;
}

export interface ProjectList {
  projects: Project[];
  organizations: Name[];
}

/** the projects the credential reaches; `workspace.get` adds their organizations */
export interface Projects {
  projects: Project[];
}

export interface ProjectsReply extends Envelope {
  result: Projects;
}

export interface ProjectListReply extends Envelope {
  result: ProjectList;
}

export interface ProjectCreate {
  name: Name;
  organization: Name;
}

export interface ProjectCreateRequest extends Envelope {
  payload: ProjectCreate;
}

/**
 * A project permission a key or role holds. Fine-tuning's: `snapshots:read` reads
 * snapshots, `snapshots:write` uploads and deletes them, `runs:read` reads runs and
 * their lineage, and `runs:write` creates and cancels runs; `models:read` reads the
 * fine-tunes runs publish. The admin and editor roles hold all four; the viewer role
 * holds `snapshots:read` and `runs:read`. `sources:manage` adds, tests, rotates and
 * deletes data sources and prepares the workspace's identity for them
 * (`sources.prepare`); only the admin role holds it, so only an admin can create a key
 * that carries it. `GET /auth/status` lists every permission with the roles that hold
 * it. The service may grant permissions this list does not name yet, so a client keeps
 * any `<group>:<action>` name it does not know as it is.
 */
export type Permission = "predictors:read" | "predictors:write" | "predictors:predict" | "contexts:read" | "contexts:write" | "contexts:history" | "sources:read" | "sources:write" | "sources:manage" | "models:read" | "snapshots:read" | "snapshots:write" | "runs:read" | "runs:write" | "keys:read" | "keys:write" | "secrets:read" | "secrets:write" | "members:read" | "members:write" | "sso:read" | "sso:write" | "audit:read" | "operators:read" | "operators:write" | "projects:read" | string;

/**
 * the signed-in account behind a key: `issuer` and `subject` name it, and stay the same
 * when its email changes
 */
export interface KeyOwner {
  issuer: string;
  subject: string;
  email: string | null;
}

/** `token` is present only in the create and rotate replies */
export interface Key {
  id: string;
  project: Name;
  name: Name;
  permissions: Permission[];
  created_at: number;
  expires_at: number | null;
  revoked_at: number | null;
  state?: "active" | "expired" | "revoked";
  prefix?: string | null;
  last_used_at?: number | null;
  ephemeral?: boolean;
  owner?: KeyOwner | null;
  service?: boolean;
  legacy?: boolean;
  machine?: string | null;
  token?: string;
  replaces?: string;
}

export interface KeyReply extends Envelope {
  result: Key;
}

export interface KeyList {
  keys: Key[];
}

export interface KeyListReply extends Envelope {
  result: KeyList;
}

export interface KeyCreate {
  name: Name;
  permissions: Permission[];
  expires_in?: number | null;
}

export interface KeyCreateRequest extends Envelope {
  payload: KeyCreate;
}

export interface KeyRef {
  id: string;
}

export interface KeyRefRequest extends Envelope {
  payload: KeyRef;
}

export interface KeyRevoked {
  id: string;
  revoked: true;
}

export interface KeyRevokedReply extends Envelope {
  result: KeyRevoked;
}

/** metadata only; a value is never returned */
export interface Secret {
  project: Name;
  name: Name;
  version: number;
  created_at: number;
  updated_at: number;
}

export interface SecretReply extends Envelope {
  result: Secret;
}

export interface SecretList {
  secrets: Secret[];
}

export interface SecretListReply extends Envelope {
  result: SecretList;
}

export interface SecretCreate {
  name: Name;
  value: string;
}

export interface SecretCreateRequest extends Envelope {
  payload: SecretCreate;
}

export interface SecretRotate {
  name: Name;
  value: string;
  expected_version: number;
}

export interface SecretRotateRequest extends Envelope {
  payload: SecretRotate;
}

export interface SecretRef {
  name: Name;
}

export interface SecretRefRequest extends Envelope {
  payload: SecretRef;
}

export interface SecretDeleted {
  name: Name;
  deleted: true;
}

export interface SecretDeletedReply extends Envelope {
  result: SecretDeleted;
}

export type RunState = "queued" | "setting_up" | "training" | "preempted" | "finished" | "cancelling" | "cancelled" | "failed";

/**
 * a snapshot a run reads, by its name and the digest the name had when the run was
 * created
 */
export interface RunData {
  name: Name;
  digest: Digest;
  phase?: Name;
  context?: Name;
  version?: number;
}

/** a component full fine-tuning may train */
export type RunComponent = "column_encoder" | "row_encoder" | "in_context_transformer" | "prediction_head";

/**
 * How a run changes the network: `{"type": "lora", "rank": 16}`, with `"input_adapter":
 * true` to train an input adapter in front of it; `{"type": "input_adapter"}`; or
 * `{"type": "full", "components": [...]}`, some of `column_encoder`, `row_encoder`,
 * `in_context_transformer` and `prediction_head`. A method the base does not take is
 * refused with 422.
 */
export interface RunMethod {
  type: "lora" | "input_adapter" | "full";
  rank?: number;
  input_adapter?: boolean;
  components?: RunComponent[];
}

/**
 * One change to the base's architecture, as a create sends it: `{"component":
 * "attention.query_scale", "stages": ["icl"], "to": {"type": "gated_log_scale"},
 * "init": {"type": "normal", "std": 0.02}}`. `stages` left out is every stage of the
 * base that can take the option; `init` left out is `{"type": "default"}`, and an
 * option without parameters (`none`, `pre`, `all`, `grouped`, `rope`) takes none.
 */
export interface RunArchitectureChange {
  component: RunArchitectureComponent;
  stages?: string[];
  to: RunArchitectureOption;
  init?: RunArchitectureInit;
}

/**
 * Something about the run to know, which does not stop it. `uncalibrated_architecture`:
 * the run changes the base's architecture, so the changed model is not calibrated to
 * the base and starts from `start_loss` where the base scores `base_loss`, on the
 * evaluation rows; both are null until step 0 measures them.
 */
export interface RunWarning {
  code: "uncalibrated_architecture";
  message: string;
  base_loss: number | null;
  start_loss: number | null;
}

/**
 * a batch's name: a snapshot name of at most 54 characters, so its tag
 * `synthetic-<name>` is a tag and its tables' names `<name>-<i>` are names
 */
export type SyntheticBatchName = string;

/**
 * the synthetic batches a run waits for before it trains, and how many of their tables
 * are still pending. `data` holds the snapshots already pinned, and the batches' ready
 * tables join it once every batch is done; absent once the run is not waiting.
 */
export interface RunWaiting {
  batches: SyntheticBatchName[];
  pending: number;
}

/** a label to find runs or snapshots by, as `recipe:fraud-v1` */
export type RunTag = string;

/**
 * every ready snapshot in the project carrying `tag`, sorted by name. The run expands
 * it when it is created, or once the synthetic batches it names are done, and keeps the
 * names and digests it found, so `runs.get` shows exactly what it trains on and tagging
 * snapshots later changes nothing. A tag no ready or generating snapshot carries is
 * refused with 422.
 */
export interface RunDataTag {
  tag: RunTag;
}

/**
 * a context version a run trains on: `version`, or the latest when it is left out,
 * pinned to that version's snapshot digest when the run is created. Its rows train to
 * predict the run's `target`
 */
export interface RunDataContext {
  context: Name;
  version?: number;
}

/**
 * what a run or phase trains on: a snapshot by its name, every ready snapshot carrying
 * a tag, or a context version. A bare name resolves as a snapshot, a synthetic batch or
 * a context, whichever has it; a name two of them have is refused with 422
 * `AmbiguousReference`, and `{"context": ...}` names a context alone
 */
export type RunDataItem = Name | RunDataTag | RunDataContext;

/**
 * where a resource was made: the surface from the request's `X-GF-Origin` header (`api`
 * when it sends none) and who made it
 */
export interface Origin {
  surface: "console" | "cli" | "notebook" | "script" | "api";
  notebook: string | null;
  client: string | null;
  by: string | null;
}

/** how the trainer draws an episode from the table's training rows */
export interface EpisodeSampling {
  rows?: number | "auto";
  context_fraction?: number;
}

/**
 * A phase's own settings: its `steps`, how it samples episodes and its episodes per
 * step, `{"steps": 400, "episode_sampling": {"rows": 1024}}`. A setting left out takes
 * the service's default, never the phase before's. Every other plan setting is set for
 * the whole run, in the run's `plan` (`RunCurriculumPlan`), and is refused here with
 * 422.
 */
export interface RunPhasePlan {
  steps: number;
  episode_sampling?: EpisodeSampling;
  episodes_per_step?: number;
}

/** a curriculum run's phase as the run keeps it, each snapshot pinned to its digest */
export interface RunPinnedPhase {
  name: Name;
  data: RunData[];
  plan: RunPhasePlan;
}

/** one phase's training so far */
export interface RunPhaseProgress {
  name: Name;
  step: number;
  loss: number | null;
}

/**
 * Where training is: the steps trained over the whole run and the latest training loss.
 * A curriculum run adds the phase it is in, by name and by its place in `phases` (from
 * 0), the steps trained within that phase, and each phase reached so far with its own
 * steps and latest training loss.
 */
export interface RunProgress {
  step: number;
  loss: number | null;
  phase?: Name;
  phase_index?: number;
  phase_step?: number;
  phases?: RunPhaseProgress[];
}

/** how far above 0 a step's paired gain over the base must be to keep it */
export interface EarlyStoppingThreshold {
  standard_errors: number;
}

/**
 * How the run holds out rows and keeps a step. `holdout` of the table's rows, clamped
 * to 2,000..50,000 rows, are held out by its time column, its group column, or
 * stratified by class or target; `report_share` of them are report rows, the rest
 * select rows. Every `eval_steps`, a check scores every select row against `contexts`
 * fixed contexts of `context_rows` training rows, the base's trained row cap when null,
 * and a step becomes the new best only if its gain over the base, row by row, is more
 * than `early_stopping_threshold.standard_errors` standard errors above 0 and beats the
 * best so far. Training stops early once the plan's `early_stopping_patience` checks in
 * a row have not lowered the run's lowest select loss, significant or not, so a run
 * whose gain is still rising goes on. A create may leave any setting out; the run says
 * each.
 */
export interface RunEvaluation {
  holdout: number;
  context_rows: number | null;
  contexts: number;
  early_stopping_threshold: EarlyStoppingThreshold;
  report_share: number;
}

/**
 * How a run trains: its `steps`, how it samples episodes, the optimizer's settings and
 * its checkpoint settings, `{"steps": 200, "learning_rate": 5e-5, "episode_sampling":
 * {"rows": 1024}}`. A setting the trainer does not read is refused with 422.
 */
export interface RunPlan {
  steps: number;
  episode_sampling?: EpisodeSampling;
  learning_rate?: number;
  optimizer?: "adamw";
  schedule?: "cosine";
  warmup_steps?: number;
  weight_decay?: number;
  beta2?: number;
  grad_clip?: number;
  episodes_per_step?: number;
  eval_steps?: number;
  early_stopping_patience?: number;
  save_steps?: number;
  save_total_limit?: number;
  save_only_model?: boolean;
}

/**
 * The settings of a curriculum run that span the whole run: the optimizer's, the
 * learning-rate schedule's, the checks' and the checkpoints', `{"learning_rate": 1e-4,
 * "warmup_steps": 20, "eval_steps": 25}`. One optimizer and one schedule run over the
 * phases' steps together, and the trained weights and the optimizer's moments carry
 * from one phase into the next. Checks, `early_stopping_patience` and the kept step
 * belong to the last phase alone, from its first step; checkpoints are saved in every
 * phase. `steps`, `episode_sampling` and `episodes_per_step` are set per phase
 * (`RunPhasePlan`) and are refused here with 422.
 */
export interface RunCurriculumPlan {
  learning_rate?: number;
  optimizer?: "adamw";
  schedule?: "cosine";
  warmup_steps?: number;
  weight_decay?: number;
  beta2?: number;
  grad_clip?: number;
  eval_steps?: number;
  early_stopping_patience?: number;
  save_steps?: number;
  save_total_limit?: number;
  save_only_model?: boolean;
}

export interface RunUsage {
  gpu_seconds: number;
  charged_gpu_seconds: number;
  voided_gpu_seconds: number;
  credits: number;
}

/**
 * the task metrics of the base's and the kept step's predictions on the report rows
 */
export interface RunReportMetrics {
  base: Record<string, number>;
  kept: Record<string, number>;
}

/**
 * Report rows, scored once for the base and once for the kept step against the check
 * contexts: their mean losses, the gain (the base's loss less the kept step's, above 0
 * when the kept step is better) with its 95% interval from the per-row gains, and task
 * metrics for each: log loss, accuracy and, for two classes, ROC AUC; or CRPS, RMSE,
 * MAE and R².
 */
export interface RunTableReport {
  rows: number;
  base: number;
  kept: number;
  gain: number;
  ci95: number[];
  metrics: RunReportMetrics;
}

/**
 * the report rows' scores under `data`, the report rows of every snapshot the run holds
 * rows out from (a curriculum run's last phase's) scored together
 */
export interface RunReport {
  data: RunTableReport;
}

/**
 * What a run published: the version, its weights' digest, the step the weights were
 * kept at and how they scored on the rows the run held out, against the base on the
 * same rows. The same as the lineage's `published` note. A version kept at step 0
 * predicts exactly as its base does: no check during training beat the base.
 */
export interface RunPublished {
  model: ModelId;
  digest: Digest;
  kept_step: KeptStep;
  validation_loss: number | null;
  base_validation_loss: number | null;
  diverged_at_step?: number;
  check_context_rows?: number;
  check_contexts?: number;
  report?: RunReport | null;
}

export interface Run {
  origin: Origin;
  url: string;
  model: RunModel;
  name: Name;
  version: number;
  state: RunState;
  reason: string | null;
  attempt: number;
  base: ModelId;
  data: RunData[];
  target?: string;
  data_digest: Digest | null;
  waiting?: RunWaiting;
  phases?: RunPinnedPhase[];
  architecture?: RunResolvedArchitectureChange[];
  warnings?: RunWarning[];
  progress: RunProgress | null;
  method: RunMethod;
  evaluation: RunEvaluation;
  budget: Record<string, unknown> | null;
  seed: number;
  tags: RunTag[];
  plan: RunPlan | RunCurriculumPlan;
  created: number;
  updated: number;
  usage?: RunUsage | null;
  published: RunPublished | null;
}

/**
 * One phase of a curriculum run, as a create sends it: its `name`, unique in the run;
 * the ready snapshots it trains on, unique in the phase, each its own table with its
 * own encoding and holdout; and its `plan` (`RunPhasePlan`), its steps and how it
 * samples episodes.
 */
export interface RunPhase {
  name: Name;
  data: RunDataItem[];
  plan: RunPhasePlan;
}

/**
 * a create's evaluation settings; each left out takes its default (`RunEvaluation`)
 */
export interface RunEvaluationRequest {
  holdout?: number;
  context_rows?: number;
  contexts?: number;
  early_stopping_threshold?: EarlyStoppingThreshold;
  report_share?: number;
}

export interface RunReply extends Envelope {
  result: Run;
}

export interface RunList {
  runs: Run[];
}

export interface RunListReply extends Envelope {
  result: RunList;
}

/**
 * A run's settings. It trains on `data` with a `plan` (`RunPlan`), or on a curriculum
 * of `phases`, with an optional `plan` of the settings that span the whole run
 * (`RunCurriculumPlan`); a create that sends both `data` and `phases`, or neither, is
 * refused with 422. `method`, `evaluation`, `budget`, `seed` and `tags` apply to the
 * whole run either way. Every snapshot is its own table, never pooled with another, and
 * has the base's task. With an input adapter every snapshot has exactly the last
 * phase's columns in the same order (with `data`, every snapshot has the same columns).
 */
export interface RunCreate {
  name: Name;
  base: ModelField;
  data?: RunDataItem[];
  target?: string;
  method: RunMethod;
  plan?: RunPlan | RunCurriculumPlan;
  phases?: RunPhase[];
  architecture?: RunArchitectureChange[];
  evaluation?: RunEvaluationRequest;
  budget?: Record<string, unknown>;
  seed?: number;
  tags?: RunTag[];
}

export interface RunCreateRequest extends Envelope {
  payload: RunCreate;
}

/** how a run is cancelled; an empty payload cancels without saving */
export interface RunCancel {
  save?: boolean;
}

export interface RunCancelRequest extends Envelope {
  payload: RunCancel;
}

export interface RunEvent {
  at: number;
  kind: string;
  body: Record<string, unknown>;
}

export interface RunLineage {
  model: RunModel;
  events: RunEvent[];
}

export interface RunLineageReply extends Envelope {
  result: RunLineage;
}

/**
 * `open` while an upload waits for its bytes, `pending` while a synthetic table is
 * generated, then `ready`, or `rejected` with an `error`; an upload left open
 * `expired`, and a snapshot `deleted`
 */
export type SnapshotState = "open" | "pending" | "ready" | "rejected" | "expired" | "deleted";

export interface SnapshotColumn {
  name: string;
  type: string;
}

/** a table someone uploaded or streamed; an upload's origin when it sends none */
export interface UploadedOrigin {
  kind: "uploaded";
}

/** a table made from a live context's rows, as they stood at `as_of` */
export interface ContextOrigin {
  kind: "context";
  context: Name;
  as_of: string | number | null;
}

/**
 * A table a synthetic batch generated: the batch, the seed the table was drawn from,
 * the generator's version, and every option as this table used it, with each range
 * replaced by its draw and `schema_from` expanded to the schema it copied, with the
 * source snapshot's name and digest. The service sets it; an upload that sends it is
 * refused with 422.
 */
export interface SyntheticOrigin {
  kind: "synthetic";
  batch: SyntheticBatchName;
  seed: number;
  generator: string;
  options: Record<string, unknown>;
}

/**
 * A table's training metadata: its `target` and `task`, and optionally its `time` and
 * `group` columns, its column types and its origin. A run refuses a snapshot without a
 * target and task. The columns set how a run holds rows out of the snapshot: its latest
 * rows with `time`, otherwise whole groups with `group`, otherwise a share stratified
 * by class or target.
 */
export interface SnapshotMetadata {
  target: string;
  task: "classification" | "regression";
  column_types?: Record<string, "numerical" | "categorical">;
  time?: string;
  group?: string;
  origin?: UploadedOrigin | ContextOrigin | SyntheticOrigin;
}

/**
 * A named, immutable table. `digest`, `rows` and `columns` are set once it is `ready`;
 * `origin` says where it was uploaded from. A create that opens an upload also carries
 * where its bytes go: `url`, `method` and `chunk_bytes`.
 */
export interface Snapshot {
  origin?: Origin;
  name: Name;
  state: SnapshotState;
  digest: Digest | null;
  bytes: number;
  rows: number | null;
  columns: SnapshotColumn[] | null;
  metadata: SnapshotMetadata | null;
  tags: RunTag[];
  error: string | null;
  expires: number | null;
  created: number;
  url?: string | null;
  method?: "PUT";
  chunk_bytes?: number;
}

export interface SnapshotReply extends Envelope {
  result: Snapshot;
}

export interface SnapshotList {
  snapshots: Snapshot[];
}

export interface SnapshotListReply extends Envelope {
  result: SnapshotList;
}

export interface SnapshotCreate {
  name: Name;
  bytes: number;
  format: "parquet";
  metadata: SnapshotMetadata;
  tags?: RunTag[];
}

export interface SnapshotCreateRequest extends Envelope {
  payload: SnapshotCreate;
}

/**
 * A snapshot's new tags, which replace the tags a user gave it; an empty list clears
 * them. The reserved tags (`uploaded`, `synthetic` and any starting `synthetic-`) stay
 * as they are and are refused here with 422. Nothing else about a snapshot changes, so
 * any other field is refused with 422.
 */
export interface SnapshotUpdate {
  tags: RunTag[];
}

export interface SnapshotUpdateRequest extends Envelope {
  payload: SnapshotUpdate;
}

/**
 * a batch is `pending` while any of its tables is, then `done`, each table `ready` or
 * `rejected`
 */
export type SyntheticBatchState = "pending" | "done";

/**
 * a kind of function a node of a synthetic table's causal graph applies to its parents
 */
export type SyntheticFunction = "linear" | "mlp" | "tree" | "gp" | "discretize" | "quadratic" | "cluster";

/** how a synthetic table's missing cells fall */
export type SyntheticMissingPattern = "random" | "rows" | "blocks" | "by_target" | "censored" | "latent" | "near_empty";

/** a range of rows per table, drawn log-uniformly */
export interface SyntheticRowsRange {
  min: number;
  max: number;
}

/** a range of feature columns per table */
export interface SyntheticColumnsRange {
  min: number;
  max: number;
}

/** a range of classes per table */
export interface SyntheticClassesRange {
  min: number;
  max: number;
}

/**
 * a range for the most categories a categorical column may have, drawn log-uniformly
 */
export interface SyntheticCategoriesRange {
  min: number;
  max: number;
}

/** a range of hidden nodes in a table's causal graph, drawn log-uniformly */
export interface SyntheticGraphNodesRange {
  min: number;
  max: number;
}

/** a range of groups of correlated columns per table */
export interface SyntheticGroupsRange {
  min: number;
  max: number;
}

/** a range of levels a coarsened column is cut into */
export interface SyntheticLevelsRange {
  min: number;
  max: number;
}

/** a range of high-cardinality categorical columns per table */
export interface SyntheticHighCardinalityRange {
  min: number;
  max: number;
}

/** a range of shares from 0 to 1 */
export interface SyntheticShareRange {
  min: number;
  max: number;
}

/** a range of shares from 0 to 0.9 */
export interface SyntheticShare90Range {
  min: number;
  max: number;
}

/** a range of shares from 0 to 0.5 */
export interface SyntheticShare50Range {
  min: number;
  max: number;
}

/**
 * how many of a table's feature columns are categorical, and how many categories each
 * may have
 */
export interface SyntheticCategorical {
  share?: number | SyntheticShareRange;
  max_categories?: number | SyntheticCategoriesRange;
}

/** groups of feature columns made to move together through a shared hidden factor */
export interface SyntheticCorrelated {
  share?: number | SyntheticShareRange;
  groups?: number | SyntheticGroupsRange;
  strength?: number | SyntheticShareRange;
}

/**
 * numeric columns cut into a few ordered levels, as a rounded or bucketed measurement
 * is
 */
export interface SyntheticCoarsened {
  share?: number | SyntheticShareRange;
  levels?: number | SyntheticLevelsRange;
}

/** missing cells, null in categorical columns and NaN in numeric ones */
export interface SyntheticMissing {
  rate?: number | SyntheticShare90Range;
  patterns?: SyntheticMissingPattern[];
}

/** how noisy, imbalanced or heavy-tailed the target `y` is */
export interface SyntheticTarget {
  noise?: number | SyntheticShareRange;
  label_noise?: number | SyntheticShare50Range;
  imbalance?: number | SyntheticShareRange;
  heavy_tail?: number | SyntheticShareRange;
}

/**
 * A synthetic batch: its name, how many tables, the seed, and the options every table
 * is drawn with. Each numeric option is an exact value or a range `{"min": a, "max":
 * b}`, a <= b, both within the option's bounds, drawn once per table; an option left
 * out takes its default. `task` is required without `schema_from`.
 */
export interface SyntheticBatchCreate {
  name: SyntheticBatchName;
  count?: number;
  seed?: number;
  schema_from?: Name;
  task?: "classification" | "regression";
  rows?: number | SyntheticRowsRange;
  columns?: number | SyntheticColumnsRange;
  classes?: number | SyntheticClassesRange;
  categorical?: SyntheticCategorical;
  functions?: SyntheticFunction[];
  graph_nodes?: number | SyntheticGraphNodesRange;
  irrelevant_columns?: number | SyntheticShare90Range;
  correlated_columns?: SyntheticCorrelated;
  coarsened_columns?: SyntheticCoarsened;
  high_cardinality_columns?: number | SyntheticHighCardinalityRange;
  missing?: SyntheticMissing;
  target?: SyntheticTarget;
  reject?: Array<"trivial" | "unlearnable">;
  tags?: RunTag[];
}

export interface SyntheticBatchCreateRequest extends Envelope {
  payload: SyntheticBatchCreate;
}

/** a batch as its create answers it, every table pending */
export interface SyntheticBatchCreated {
  name: SyntheticBatchName;
  tag: RunTag;
  count: number;
  snapshots: Name[];
  state: SyntheticBatchState;
  created: number;
  origin: Origin;
  url: string;
}

/**
 * one table of a batch, by its snapshot name, with its state and, once rejected, why
 */
export interface SyntheticTable {
  name: Name;
  state: SnapshotState;
  error: string | null;
}

/** how many of a batch's tables are in each state */
export interface SyntheticCounts {
  pending: number;
  ready: number;
  rejected: number;
}

/**
 * a batch as `snapshots.synthetic` reads it: its create's answer, and the seed,
 * options, tables and counts. Its fields are listed whole, as `SyntheticBatchCreated`
 * allows no other fields.
 */
export interface SyntheticBatch {
  name: SyntheticBatchName;
  tag: RunTag;
  count: number;
  snapshots: Name[];
  state: SyntheticBatchState;
  created: number;
  origin: Origin;
  url: string;
  seed: number;
  options: SyntheticBatchCreate;
  tables: SyntheticTable[];
  counts: SyntheticCounts;
}

export interface SyntheticBatchCreatedReply extends Envelope {
  result: SyntheticBatchCreated;
}

export interface SyntheticBatchReply extends Envelope {
  result: SyntheticBatch;
}

export type State = "pending" | "building" | "ready" | "failed" | "stopped";

/** a predictor's spec: `gf.Predictor`'s fields but its name, as JSON */
export interface Spec {
  context: string;
  model: ModelField;
  target: string;
  task?: Task;
  id?: string;
  time?: string;
  inputs?: string[];
  exclude?: string[];
  types?: Record<string, ColumnType>;
  series?: string[];
  frequency?: string | number;
  horizon?: number;
  quantiles?: number[];
  past?: string[];
  future?: string[];
  selection?: Selection;
}

/**
 * what a spec resolved to against its context version's rows, by the engine's
 * `resolve`: the full model id and digest, the task, inputs and their types. A deploy
 * compares these, so a short name and its full id are one predictor
 */
export interface Resolved {
  model: ModelField;
  short: string | null;
  model_digest: Digest;
  base_digest: Digest | null;
  task: Task;
  context: Name;
  context_version: number | null;
  context_digest: Digest | null;
  context_rows?: number | null;
  inputs: string[];
  types: Record<string, ColumnType>;
}

/** an immutable spec with what it resolved to; the plane keeps its lock beside it */
export interface Revision {
  predictor: Name;
  number: number;
  status: "pending" | "building" | "ready" | "failed" | "stopped";
  lock_digest: Digest;
  spec: Spec;
  resolved: Resolved;
  origin: Origin;
  created: string;
}

export interface RevisionReply extends Envelope {
  result: Revision;
}

export interface RevisionList {
  revisions: Revision[];
}

export interface RevisionListReply extends Envelope {
  result: RevisionList;
}

/**
 * why an environment's revision failed to start: the failure in words, the request
 * whose deploy or promote started it, and the revision the environment still serves
 * (null when it served none)
 */
export interface EnvironmentFailure {
  message: string;
  request_id: string | null;
  serving_revision: number | null;
}

/** a pointer to a revision; `generation` guards promote and rollback */
export interface Environment {
  name: Name;
  revision: number | null;
  state: "starting" | "ready" | "failed" | "stopped";
  generation: number;
  resolved: Resolved | null;
  calls_today?: number;
  p50_seconds?: number | null;
  updated?: string;
  url: string;
  error?: EnvironmentFailure | null;
}

export interface EnvironmentReply extends Envelope {
  result: Environment;
}

export interface EnvironmentList {
  environments: Environment[];
}

export interface EnvironmentListReply extends Envelope {
  result: EnvironmentList;
}

/**
 * the revision to point at, or `from_env`, the environment whose revision to take;
 * `expected_generation` refuses a change past a stale read
 */
export interface Promote {
  revision?: number;
  from_env?: Name;
  expected_generation?: number;
}

export interface PromoteRequest extends Envelope {
  payload: Promote;
}

/**
 * the context an environment serves: `source` is `snapshot`, `request` or
 * `live/<context>`; the version fields and `freshness_lag_seconds` are set for live
 * contexts
 */
export interface ServingContext {
  source: string;
  version: number | string | null;
  rows: number;
  as_of: AsOf;
  accepted_version: number | string | null;
  serving_version: number | string | null;
  freshness_lag_seconds: number | null;
  [key: string]: unknown;
}

/**
 * serving facts: `context` is the context the environment is serving now,
 * `accepted_version`/`serving_version` compare the newest accepted write of a live
 * context with the one being served, `fresh` says whether they agree (null without a
 * context); `error` is set while the environment serves its last good view and a
 * rebuild of it keeps failing, else null
 */
export interface Metrics {
  status: State;
  error: string | null;
  context: ServingContext | null;
  accepted_version: number | string | null;
  serving_version: number | string | null;
  fresh: boolean | null;
  requests: number;
  errors: number;
  mean_latency_seconds: number | null;
}

export interface MetricsReply extends Envelope {
  result: Metrics;
}

/** an entry's place in its predictor's log, as a reply gave it */
export type LogCursor = string;

export type LogLevel = "info" | "warn" | "error";

/**
 * One thing that happened. `id` is the entry's cursor, which `after` and `before` take.
 */
export interface LogEntry {
  id: LogCursor;
  at: number;
  level: LogLevel;
  source: string;
  code: string;
  message: string;
  environment: string | null;
  revision: number | null;
  request_id: string | null;
  fields: Record<string, string | number | boolean>;
}

/**
 * A page of a log, oldest first. `next` is the cursor to pass as `after` for the
 * entries that follow the page.
 */
export interface Logs {
  logs: LogEntry[];
  next: LogCursor;
  previous: string | null;
}

export interface LogsReply extends Envelope {
  result: Logs;
}

export interface LogBucket {
  at: number;
  info: number;
  warn: number;
  error: number;
}

export interface LogHistogram {
  since: number;
  until: number;
  bucket_seconds: number;
  buckets: LogBucket[];
}

export interface LogHistogramReply extends Envelope {
  result: LogHistogram;
}

export interface PredictorDeleted {
  name: Name;
  deleted: true;
}

export interface PredictorDeletedReply extends Envelope {
  result: PredictorDeleted;
}

export interface LiveContextAppend {
  write_id: string;
  rows: Row[];
}

export interface LiveContextAppendRequest extends Envelope {
  payload: LiveContextAppend;
}

export interface Label {
  id: string | number;
  value: string | number | boolean;
}

export interface LiveContextLabel {
  write_id: string;
  target: string;
  labels: Label[];
}

export interface LiveContextLabelRequest extends Envelope {
  payload: LiveContextLabel;
}

/** durable acceptance of an append or label write */
export interface Receipt {
  context: Name;
  write_id: string;
  operation: "append" | "label";
  rows: number;
  sequence: number;
  accepted_at: number;
  durably_accepted: boolean;
}

export interface ReceiptReply extends Envelope {
  result: Receipt;
}

export interface PredictorCreate {
  name: Name;
  spec: Spec;
  env: Name;
  dry_run?: boolean;
}

export interface PredictorCreateRequest extends Envelope {
  payload: PredictorCreate;
}

export interface SourcePrepare {
  kind: "bigquery" | "gcs" | "s3" | "databricks" | "snowflake";
}

export interface SourcePrepareRequest extends Envelope {
  payload: SourcePrepare;
}

/** the workspace's Google service account */
export interface SourceIdentity {
  email: string;
  unique_id: string;
  principal: string;
}

export interface SourceGrant {
  role: string;
  on: "dataset" | "bucket" | "project";
}

/**
 * what another cloud trusts: the issuer, subject and audience of the account's Google
 * ID tokens
 */
export interface SourceTrust {
  issuer: string;
  subject: string;
  audience?: string;
}

/**
 * the document the customer pastes: an AWS role's trust policy, a Databricks service
 * principal's federation policy
 */
export interface SourcePolicy {
  format: "json";
  text: string;
}

export interface SourcePrepared {
  kind: "bigquery" | "gcs" | "s3" | "databricks" | "snowflake";
  identity: SourceIdentity;
  grants: SourceGrant[];
  steps: string[];
  oidc?: SourceTrust;
  policy?: SourcePolicy;
}

export interface SourcePreparedReply extends Envelope {
  result: SourcePrepared;
}

export type SourceKind = "bigquery" | "gcs" | "snowflake" | "databricks" | "s3" | "postgres";

/**
 * a workspace connection to a warehouse or bucket. Config keys by kind: bigquery
 * `project, dataset, billing_project` (queries run and bill in billing_project, the
 * dataset's project unless named); gcs `bucket, prefix`; snowflake `account, warehouse,
 * database, role`; databricks `host, warehouse, client_id, catalog, schema` (client_id
 * is the federated service principal's application id); s3 `role_arn, region, bucket,
 * prefix`; postgres `host, port, database, user, tls`. A bucket's tables are its
 * Parquet and CSV files and folders of their parts under the prefix
 */
export interface Source {
  name: Name;
  kind: SourceKind;
  access: "service_account" | "secret";
  state: "untested" | "connected" | "unauthorized" | "not_found" | "unavailable" | "expired";
  config: Record<string, string>;
  secret_expires: string | null;
  tested_at: string | null;
  tables: number | null;
  contexts: number;
  error: string | null;
  origin: Origin;
  created: string;
  url: string;
}

/** `available` says whether this plane can connect sources at all */
export interface SourceList {
  available: boolean;
  sources: Source[];
}

export interface SourceListReply extends Envelope {
  result: SourceList;
}

/**
 * a permission a role holds; one that is not enforced authorizes nothing yet.
 * `key_grantable` says whether an API key may hold it: false for those only a signed-in
 * browser session uses (`operators:read`, `sso:read`) and for every reserved one, true
 * for every other enforced permission. `keys.create` and `keys.rotate` refuse a key a
 * permission it may not hold
 */
export interface PermissionGrant {
  name: Permission;
  description: string;
  enforced: boolean;
  key_grantable: boolean;
  roles: Array<"admin" | "editor" | "viewer" | "scorer">;
}

export interface KeyIdentity {
  id: string;
  name: Name;
  project: Name;
  expires_at: number | null;
  ephemeral?: boolean;
}

export interface WorkspaceIdentity {
  id: string;
  display_name: string;
}

/** the credential's kind and id, with what the server knows of it */
export interface Principal {
  kind: string;
  id: string;
  [key: string]: unknown;
}

/**
 * the sizes this service takes data in, for a client to size its requests by instead of
 * copying them: one streamed snapshot (`snapshots.create` with an Arrow body, `?name=`)
 * holds at most `snapshot_rows` rows and `snapshot_bytes` bytes; a resumable upload at
 * most `upload_rows` rows and `upload_bytes` bytes, or `csv_bytes` for a CSV, sent in
 * chunks of `chunk_bytes` unless the upload names its own; a JSON request body holds at
 * most `request_bytes`
 */
export interface UploadLimits {
  snapshot_bytes: number;
  snapshot_rows: number;
  upload_bytes: number;
  upload_rows: number;
  csv_bytes: number;
  chunk_bytes: number;
  request_bytes: number;
}

export interface AuthStatus {
  authenticated: true;
  principal: Principal;
  sign_in: Array<"google" | "microsoft" | "github">;
  secrets: "configured" | "unconfigured";
  capabilities: Capability[];
  permission_inventory: PermissionGrant[];
  bootstrap: "unconfigured" | "available" | "consumed";
  trains: boolean;
  limits: UploadLimits;
  key?: KeyIdentity;
  email?: string | null;
  machine?: string | null;
  workspace?: WorkspaceIdentity | null;
  url?: string;
}

export interface AuthStatusReply extends Envelope {
  result: AuthStatus;
}

export interface LoggedOut {
  revoked: string;
}

export interface LoggedOutReply extends Envelope {
  result: LoggedOut;
}

export interface Organization {
  name: Name;
  account: Name;
}

export interface OrganizationReply extends Envelope {
  result: Organization;
}

export interface OrganizationCreateRequest extends Envelope {
  payload: Organization;
}

export interface OrganizationList {
  organizations: Organization[];
}

export interface OrganizationListReply extends Envelope {
  result: OrganizationList;
}

export interface Sso {
  organization: Name;
  providers: Array<"google" | "microsoft" | "github">;
  enabled: boolean;
  configured: boolean;
  membership_required: true;
}

export interface SsoReply extends Envelope {
  result: Sso;
}

export interface SsoConfigure {
  enabled: boolean;
}

export interface SsoConfigureRequest extends Envelope {
  payload: SsoConfigure;
}

export interface AuditEvent {
  sequence: number;
  at: number;
  actor: string;
  project: Name;
  action: string;
  target: string;
  outcome: string;
}

export interface AuditLog {
  events: AuditEvent[];
}

export interface AuditLogReply extends Envelope {
  result: AuditLog;
}

export interface DeviceLoginStart {
  machine: string;
  client: "cli" | "notebook";
}

export interface DeviceLoginGrant {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete: string;
  expires_in: number;
  interval: number;
}

export interface DeviceLoginPoll {
  device_code: string;
}

export interface DeviceLoginToken {
  access_token: string;
  token_type: "bearer";
  expires_at: number | null;
  key: KeyIdentity;
  email: string;
  workspace: WorkspaceIdentity;
  project: Name;
  url: string;
}

/**
 * why a device login cannot go on, outside the envelope as the login is. `reason` says
 * which: `authorization_pending` (`Unauthorized`, poll again), `slow_down`
 * (`ValidationError`, poll less often), `expired_token` and `access_denied`
 * (`Unauthorized`, start again) or `rate_limited` (`CapacityExceeded`).
 */
export interface LoginErrorReply {
  error: ApiError;
}

/** a workspace member's role; `admin` manages the workspace */
export type WorkspaceRole = "admin" | "editor" | "viewer" | "scorer";

export type JoinRequestId = string;

export type InvitationId = string;

export type WaitlistId = string;

/** a workspace's display name, without leading or trailing spaces */
export type WorkspaceName = string;

/**
 * the code `gf login` prints, `XXXX-XXXX` from BCDFGHJKLMNPQRSTVWXZ, taken in any case
 * and with or without the dash
 */
export type UserCode = string;

/** a workspace the caller belongs to: its role there and the projects it may use */
export interface WorkspaceSummary {
  workspace: Name;
  display_name: string;
  role: WorkspaceRole | null;
  admin: boolean;
  default_project: Name | null;
  projects: Name[];
}

/**
 * a workspace as `workspaces.get` reads it: its summary, its organization and when it
 * was made. Its fields are listed whole, as `WorkspaceSummary` allows no other fields.
 */
export interface WorkspaceDetails {
  workspace: Name;
  display_name: string;
  role: WorkspaceRole | null;
  admin: boolean;
  default_project: Name | null;
  projects: Name[];
  organization: Name;
  created_at: number;
}

/** a request to join a workspace, as the person who asked sees it */
export interface JoinRequest {
  id: JoinRequestId;
  workspace: Name;
  display_name: string;
  state: "pending" | "approved" | "denied" | "cancelled";
  created_at: number;
  decided_at: number | null;
}

/** a request to join a workspace, as its admins see it */
export interface JoinRequestAdmin {
  id: JoinRequestId;
  workspace: Name;
  subject: string;
  email: string;
  name: string | null;
  domain: string;
  state: "pending" | "approved" | "denied" | "cancelled";
  role: WorkspaceRole | null;
  created_at: number;
  decided_at: number | null;
  decided_by: string | null;
}

/** the signed-in person's verified email, all null until they sign in again */
export interface OnboardingProfile {
  email: string | null;
  email_domain: string | null;
  display_name: string | null;
  personal_domain: boolean | null;
}

/** a workspace the person's email domain may ask to join */
export interface DiscoveredWorkspace {
  workspace: Name;
  display_name: string;
  domain: string;
  request: JoinRequest | null;
}

/**
 * where a signed-in person stands: their workspaces, those their domain may join, and
 * whether they may create one
 */
export interface Onboarding {
  state: "member" | "onboarding" | "reauthentication_required";
  profile: OnboardingProfile;
  workspaces: WorkspaceSummary[];
  discovered: DiscoveredWorkspace[];
  requests: JoinRequest[];
  can_create_workspace: boolean;
  signup: "open" | "waitlist" | "pending";
  workspace_name: string | null;
  sign_in_url: string;
}

export interface WorkspaceList {
  workspaces: WorkspaceSummary[];
}

export interface WorkspaceCreate {
  display_name: WorkspaceName;
}

/** `created` is false when the request id already made this workspace */
export interface WorkspaceCreated {
  workspace: WorkspaceSummary;
  organization: Name;
  created: boolean;
}

export interface WorkspaceRef {
  workspace: Name;
}

export interface WorkspaceRename {
  workspace: Name;
  display_name: WorkspaceName;
}

/**
 * `workspace` is set once the caller is a member, `request` while a join request is
 * pending or was denied lately
 */
export interface WorkspaceJoin {
  status: "already_member" | "pending" | "denied";
  workspace: WorkspaceSummary | null;
  request: JoinRequest | null;
}

/** one membership in the workspace, its organizations or projects */
export interface WorkspaceMember {
  issuer: string;
  subject: string;
  email: string | null;
  name: string | null;
  scope_kind: "account" | "organization" | "project";
  scope_id: string;
  role: WorkspaceRole;
}

export interface WorkspaceMemberList {
  members: WorkspaceMember[];
}

export interface MemberRoleChange {
  workspace: Name;
  issuer: string;
  subject: string;
  role: WorkspaceRole;
}

export interface MemberRole {
  issuer: string;
  subject: string;
  role: WorkspaceRole;
}

export interface MemberRoleChanged {
  member: MemberRole;
}

export interface MemberRemove {
  workspace: Name;
  issuer: string;
  subject: string;
}

export interface MemberGone {
  issuer: string;
  subject: string;
  removed: true;
}

export interface MemberRemoved {
  member: MemberGone;
}

/** an invitation to a workspace; `state` is `expired` once a pending one lapses */
export interface Invitation {
  id: InvitationId;
  workspace: Name;
  email: string;
  role: WorkspaceRole;
  state: "pending" | "expired" | "accepted" | "revoked";
  created_at: number;
  expires_at: number;
  created_by: string;
  accepted_at: number | null;
  revoked_at: number | null;
}

export interface InvitationList {
  invitations: Invitation[];
}

/** `role` is `viewer` and `expires_in` a week unless given */
export interface InvitationCreate {
  workspace: Name;
  email: string;
  role?: WorkspaceRole;
  expires_in?: number;
}

/**
 * `token` and `invitation_url` only when the server has no mailer and `delivery` is
 * `link`; `replaces` names a pending invitation to the same email it revoked
 */
export interface InvitationCreated {
  invitation: Invitation;
  delivery: "email" | "link";
  replaces: InvitationId | null;
  token?: string;
  invitation_url?: string;
}

export interface InvitationRevoke {
  workspace: Name;
  id: InvitationId;
}

export interface InvitationView {
  invitation: Invitation;
}

/** the token from the invitation link */
export interface InvitationToken {
  token: string;
}

export interface InvitationPreview {
  workspace: Name;
  display_name: string;
  role: WorkspaceRole;
  expires_at: number;
}

export interface InvitationAccepted {
  status: "joined" | "already_member";
  workspace: WorkspaceSummary;
}

export interface JoinRequestRef {
  id: JoinRequestId;
}

export interface JoinRequestView {
  request: JoinRequest;
}

export interface JoinRequestList {
  requests: JoinRequestAdmin[];
}

/** `role` is `viewer` unless given */
export interface JoinRequestApprove {
  workspace: Name;
  id: JoinRequestId;
  role?: WorkspaceRole;
}

export interface JoinRequestDeny {
  workspace: Name;
  id: JoinRequestId;
}

/** the account membership an approval granted, or the one the person already held */
export interface WorkspaceMembership {
  issuer: string;
  subject: string;
  scope_kind: "account";
  scope_id: Name;
  role: WorkspaceRole;
}

export interface JoinRequestApproved {
  request: JoinRequestAdmin;
  membership: WorkspaceMembership;
}

export interface JoinRequestDenied {
  request: JoinRequestAdmin;
}

/** an email domain whose people may ask to join the workspace */
export interface JoinDomain {
  workspace: Name;
  domain: string;
  enabled_by: string;
  enabled_at: number;
}

/**
 * `email_domain` is the caller's own company domain, the one it may enable; null for a
 * personal one
 */
export interface JoinDomainList {
  domains: JoinDomain[];
  email_domain: string | null;
}

/** enabling takes only the domain of the admin's own verified email */
export interface JoinDomainConfigure {
  workspace: Name;
  domain: string;
  enabled: boolean;
}

export interface JoinDomainConfigured {
  workspace: Name;
  domain: string;
  enabled: boolean;
}

/** `display_name` is the workspace the person means to create */
export interface WaitlistJoin {
  display_name?: WorkspaceName;
}

export interface WaitlistStatus {
  status: "approved" | "pending";
  message: string | null;
}

/**
 * a waitlist sign-up; approving one from a company domain approves the whole domain
 */
export interface WaitlistEntry {
  id: WaitlistId;
  email: string;
  display_name: string | null;
  state: "pending" | "approved";
  approves: "person" | "domain";
  domain: string;
  created_at: number;
  decided_at: number | null;
  workspace_name: string | null;
}

export interface WaitlistRef {
  id: WaitlistId;
}

export interface WaitlistEntryView {
  entry: WaitlistEntry;
}

export interface DeviceLoginCode {
  user_code: UserCode;
}

export interface DeviceLoginWorkspace {
  workspace: Name;
  display_name: string;
}

/**
 * what the sign-in page confirms: the machine and client asking ("Approve the CLI on
 * maya-mbp", "Approve a notebook on maya-mbp") and the workspaces with a project the
 * key may serve
 */
export interface DeviceLoginPreview {
  user_code: string;
  machine: string;
  client: "cli" | "notebook";
  expires_at: number;
  email: string;
  workspaces: DeviceLoginWorkspace[];
}

export interface DeviceLoginApprove {
  user_code: UserCode;
  workspace: Name;
}

export interface DeviceLoginGranted {
  id: Name;
  display_name: string;
}

export interface DeviceLoginApproved {
  user_code: string;
  state: "approved";
  workspace: DeviceLoginGranted;
}

export interface DeviceLoginDenied {
  user_code: string;
  state: "denied";
}

export interface DeviceLoginApproveRequest extends Envelope {
  payload: DeviceLoginApprove;
}

export interface DeviceLoginCodeRequest extends Envelope {
  payload: DeviceLoginCode;
}

export interface InvitationCreateRequest extends Envelope {
  payload: InvitationCreate;
}

export interface InvitationRevokeRequest extends Envelope {
  payload: InvitationRevoke;
}

export interface InvitationTokenRequest extends Envelope {
  payload: InvitationToken;
}

export interface JoinDomainConfigureRequest extends Envelope {
  payload: JoinDomainConfigure;
}

export interface JoinRequestApproveRequest extends Envelope {
  payload: JoinRequestApprove;
}

export interface JoinRequestDenyRequest extends Envelope {
  payload: JoinRequestDeny;
}

export interface JoinRequestRefRequest extends Envelope {
  payload: JoinRequestRef;
}

export interface MemberRemoveRequest extends Envelope {
  payload: MemberRemove;
}

export interface MemberRoleChangeRequest extends Envelope {
  payload: MemberRoleChange;
}

export interface WaitlistJoinRequest extends Envelope {
  payload: WaitlistJoin;
}

export interface WaitlistRefRequest extends Envelope {
  payload: WaitlistRef;
}

export interface WorkspaceCreateRequest extends Envelope {
  payload: WorkspaceCreate;
}

export interface WorkspaceRefRequest extends Envelope {
  payload: WorkspaceRef;
}

export interface WorkspaceRenameRequest extends Envelope {
  payload: WorkspaceRename;
}

export interface DeviceLoginApprovedReply extends Envelope {
  result: DeviceLoginApproved;
}

export interface DeviceLoginDeniedReply extends Envelope {
  result: DeviceLoginDenied;
}

export interface DeviceLoginPreviewReply extends Envelope {
  result: DeviceLoginPreview;
}

export interface InvitationAcceptedReply extends Envelope {
  result: InvitationAccepted;
}

export interface InvitationCreatedReply extends Envelope {
  result: InvitationCreated;
}

export interface InvitationListReply extends Envelope {
  result: InvitationList;
}

export interface InvitationPreviewReply extends Envelope {
  result: InvitationPreview;
}

export interface InvitationViewReply extends Envelope {
  result: InvitationView;
}

export interface JoinDomainConfiguredReply extends Envelope {
  result: JoinDomainConfigured;
}

export interface JoinDomainListReply extends Envelope {
  result: JoinDomainList;
}

export interface JoinRequestApprovedReply extends Envelope {
  result: JoinRequestApproved;
}

export interface JoinRequestDeniedReply extends Envelope {
  result: JoinRequestDenied;
}

export interface JoinRequestListReply extends Envelope {
  result: JoinRequestList;
}

export interface JoinRequestViewReply extends Envelope {
  result: JoinRequestView;
}

export interface MemberRemovedReply extends Envelope {
  result: MemberRemoved;
}

export interface MemberRoleChangedReply extends Envelope {
  result: MemberRoleChanged;
}

export interface OnboardingReply extends Envelope {
  result: Onboarding;
}

export interface WaitlistEntryViewReply extends Envelope {
  result: WaitlistEntryView;
}

export interface WaitlistStatusReply extends Envelope {
  result: WaitlistStatus;
}

export interface WorkspaceCreatedReply extends Envelope {
  result: WorkspaceCreated;
}

export interface WorkspaceDetailsReply extends Envelope {
  result: WorkspaceDetails;
}

export interface WorkspaceJoinReply extends Envelope {
  result: WorkspaceJoin;
}

export interface WorkspaceListReply extends Envelope {
  result: WorkspaceList;
}

export interface WorkspaceMemberListReply extends Envelope {
  result: WorkspaceMemberList;
}

export interface WorkspaceSummaryReply extends Envelope {
  result: WorkspaceSummary;
}

/**
 * the same resource made in code: the SDK call, the `gf` command and a curl request
 */
export interface Code {
  python: string;
  cli: string;
  curl: string;
}

export interface Deleted {
  name: Name;
  deleted: true;
}

/**
 * the large-context line: GPU seconds past `large_context_cells` in the price book, and
 * their cost
 */
export interface LargeContext {
  gpu_seconds: number;
  credits: number;
  usd: number;
}

/**
 * what a call or a run would cost from the live price book: model calls, scored rows,
 * credits (1 credit is $0.001) and dollars. `calls` is the model calls the queries of
 * each call decode in, as the engine's worker splits them by its shape buckets (the
 * engine's `gf_engine.hosting.estimate_calls`); no model bounds the queries of a call,
 * and an answered call's `timing.calls` says what it took
 */
export interface Estimate {
  calls: number;
  rows: number;
  credits: number;
  usd: number;
  large_context?: LargeContext;
  warnings?: string[];
}

export interface Column {
  name: string;
  type: ColumnType;
  missing: number;
  distinct: number | null;
}

/**
 * an upload's announcement; the bytes follow in chunks unless the rows are unchanged
 */
export interface ContextData {
  format: "parquet" | "arrow" | "csv";
  bytes: number;
  digest: Digest;
  rows?: number;
  file?: string | null;
}

/** what a context reads from a source: one `table` or a `sql` query */
export interface ContextSourceRead {
  source: Name;
  table?: string;
  sql?: string;
}

export interface ContextSource {
  source: Name;
  kind: SourceKind;
  table: string | null;
  sql: string | null;
}

/**
 * one origin: `data` (an upload, version 1), `source` (an import, or with `kind: live`
 * a read at call time cached for `freshness_seconds`), or `kind: live` with `columns`
 * for rows that arrive through `append`
 */
export interface ContextCreate {
  name: Name;
  kind: "snapshot" | "live";
  data?: ContextData;
  source?: ContextSourceRead;
  columns?: Record<string, ColumnType>;
  targets?: string[];
  id?: string | null;
  time?: string | null;
  freshness_seconds?: number;
}

export interface ContextUploadTarget {
  version: number;
  url: string | null;
  chunk_bytes: number;
}

/**
 * the data a model learns from; `version` is the latest (or the one read), `upload`
 * where an upload's chunks go
 */
export interface Context {
  name: Name;
  kind: "snapshot" | "live";
  state: "uploading" | "importing" | "ready" | "failed";
  version: number | null;
  versions: number;
  digest: Digest | null;
  rows: number | null;
  columns: Column[];
  id: string | null;
  time: string | null;
  targets?: string[];
  source: ContextSource | null;
  freshness_seconds: number | null;
  read_at: string | null;
  error: string | null;
  predictors: Name[];
  experiments: Name[];
  fine_tunes: string[];
  origin: Origin;
  created: string;
  updated: string;
  url: string;
  file?: string | null;
  upload?: ContextUploadTarget;
  unchanged?: boolean;
}

export interface ContextUpload {
  data: ContextData;
}

export interface ContextList {
  contexts: Context[];
}

export interface ContextVersionInfo {
  version: number;
  digest: Digest;
  rows: number;
  created: string;
  origin: Origin;
}

export interface ContextVersions {
  versions: ContextVersionInfo[];
}

/** a page of rows; `next` is the cursor for the page after, or null at the end */
export interface Rows {
  columns: Column[];
  rows: Row[];
  next: string | null;
}

/** the bytes of the version the service holds so far */
export interface Received {
  received: number;
}

/**
 * the rows held out to score: a random share or count (optionally stratified), whole
 * groups of a column, the latest by time, or none
 */
export interface Holdout {
  kind: "random" | "group" | "time" | "none";
  fraction?: number;
  rows?: number;
  column?: string;
  until?: string;
  stratify?: boolean;
  seed?: number;
  repeats?: number;
}

/**
 * an experiment's setup; `name` defaults to `<context>-<target>`, the task is inferred
 * from the target, the holdout is the latest 20% by `time` when one is declared, else
 * 20% at random
 */
export interface ExperimentSetup {
  name?: Name;
  title?: string;
  context: Name;
  version?: number;
  target: string;
  task?: Task;
  id?: string;
  time?: string;
  inputs?: string[];
  exclude?: string[];
  types?: Record<string, ColumnType>;
  holdout?: Holdout;
  sql?: string;
}

/** the setup, and the models of run 1: the recommended set when left out */
export interface ExperimentCreate {
  name?: Name;
  title?: string;
  context: Name;
  version?: number;
  target: string;
  task?: Task;
  id?: string;
  time?: string;
  inputs?: string[];
  exclude?: string[];
  types?: Record<string, ColumnType>;
  holdout?: Holdout;
  sql?: string;
  models?: ModelField[];
  dry_run?: boolean;
}

/** the setup fields that change; earlier runs become `stale` */
export interface ExperimentUpdate {
  title?: string;
  context?: Name;
  version?: number;
  target?: string;
  task?: Task;
  id?: string;
  time?: string;
  inputs?: string[];
  exclude?: string[];
  types?: Record<string, ColumnType>;
  holdout?: Holdout;
  sql?: string;
}

export interface Entry {
  model: ModelField;
  short: string | null;
  state: "queued" | "running" | "finished" | "failed" | "cancelled";
  score: number | null;
  interval: number[] | null;
  metrics: Record<string, number>;
  seconds: number;
  rows: number;
  credits: number;
  usd: number;
  error: string | null;
}

/**
 * a run: its models scored on the held-out rows, best first; it pins the context
 * digest, setup digest and model digests, and is `stale` once the setup moved
 */
export interface ExperimentRun {
  experiment: Name;
  number: number;
  state: "queued" | "running" | "finished" | "failed" | "cancelled";
  stale: boolean;
  models: ModelId[];
  metric: string;
  lower_is_better: boolean;
  held_rows: number;
  context_rows: number;
  repeats: number;
  leaderboard: Entry[];
  rows: number;
  credits: number;
  usd: number;
  created: string;
  started: string | null;
  finished: string | null;
  origin: Origin;
  error: string | null;
  url: string;
}

export interface ExperimentRunSummary {
  number: number;
  state: "queued" | "running" | "finished" | "failed" | "cancelled";
  models: ModelId[];
  best: string | null;
  score: number | null;
  origin: Origin;
  created: string;
}

/**
 * models evaluated on one context for one target; `run` is the latest, or the one asked
 * for
 */
export interface Experiment {
  name: Name;
  title: string;
  context: Name;
  context_version: number;
  target: string;
  task: Task;
  id: string | null;
  time: string | null;
  inputs: string[];
  types: Record<string, ColumnType>;
  holdout: Holdout;
  sql: string | null;
  metric: string;
  runs: ExperimentRunSummary[];
  run: ExperimentRun | null;
  origin: Origin;
  created: string;
  updated: string;
  url: string;
}

/** a dry run: the setup as it resolves, and what run 1 would cost */
export interface ExperimentPlan {
  setup: ExperimentSetup;
  estimate: Estimate;
  warnings: string[];
}

/** the experiment with run 1 queued, or under `dry_run` its plan */
export type ExperimentCreated = Experiment | ExperimentPlan;

/** the experiments, each without its run's leaderboard */
export interface ExperimentList {
  experiments: Experiment[];
  next: string | null;
}

export interface ExperimentRunCreate {
  models?: ModelField[];
  dry_run?: boolean;
}

export interface ExperimentRunEstimate {
  estimate: Estimate;
}

export type ExperimentRunCreated = ExperimentRun | ExperimentRunEstimate;

export interface ExperimentRunList {
  runs: ExperimentRun[];
}

export interface Slice {
  column: string;
  value: string;
  rows: number;
  score: number;
  delta: number;
}

export interface SliceList {
  slices: Slice[];
}

/** the held-out rows with one model's predictions beside them */
export interface HeldRows {
  model: ModelField;
  rows: Row[];
  next: string | null;
}

/** the spec fields that change; `null` clears one */
export interface SpecPatch {
  context?: string | null;
  model?: ModelField | null;
  target?: string | null;
  task?: Task | null;
  id?: string | null;
  time?: string | null;
  inputs?: string[] | null;
  exclude?: string[] | null;
  types?: Record<string, ColumnType> | null;
  series?: string[] | null;
  frequency?: string | number | null;
  horizon?: number | null;
  quantiles?: number[] | null;
  past?: string[] | null;
  future?: string[] | null;
  selection?: Selection | null;
}

export interface PredictorUpdate {
  spec: SpecPatch;
  env: Name;
  dry_run?: boolean;
}

export interface PredictorDeploy {
  spec: Spec;
  env: Name;
  dry_run?: boolean;
}

export interface Change {
  field: string;
  before: string | null;
  after: string | null;
}

/**
 * what a create, update or deploy did; `unchanged` wrote nothing. The environment is
 * `starting` until its replica is ready, which `environments.get` reports
 */
export interface Deploy {
  predictor: Name;
  env: Name;
  action: "created" | "updated" | "unchanged";
  changes: Change[];
  revision: Revision;
  environment: Environment | null;
}

/** a named, versioned endpoint: its latest revision's spec and each environment */
export interface PredictorInfo {
  name: Name;
  spec: Spec;
  resolved: Resolved;
  revision: number;
  environments: Record<string, Environment>;
  origin: Origin;
  created: string;
  url: string;
}

export interface PredictorInfoList {
  predictors: PredictorInfo[];
}

export interface EstimateQuery {
  model: ModelField;
  task?: Task;
  context_rows: number;
  queries: number;
  inputs?: number;
}

export interface SourceCreate {
  name: Name;
  kind: SourceKind;
  config: Record<string, string>;
  secret?: string;
}

/** a table of the source's location, as the warehouse lists it */
export interface SourceTable {
  name: string;
  type: string;
  readable: boolean;
}

/**
 * the tables of the source's location, whose names start with `prefix`; `location` is
 * where they are, as a person writes it (`project.dataset`, `gs://bucket/prefix/`,
 * `s3://bucket/prefix/`, `catalog.schema`); `more` says the location holds more than
 * were listed
 */
export interface SourceTables {
  location: string;
  tables: SourceTable[];
  more: boolean;
}

/** one `table` or one `sql` query; the answer is its first 20 rows */
export interface SourcePreview {
  table?: string;
  sql?: string;
}

export interface KeyDeleted {
  id: string;
  deleted: true;
}

export interface ContextReply extends Envelope {
  result: Context;
}

export interface ContextListReply extends Envelope {
  result: ContextList;
}

export interface ContextVersionsReply extends Envelope {
  result: ContextVersions;
}

export interface RowsReply extends Envelope {
  result: Rows;
}

export interface ReceivedReply extends Envelope {
  result: Received;
}

/**
 * the bytes of a snapshot's upload the plane holds so far, of the `bytes` it declared
 */
export interface SnapshotReceived {
  name: Name;
  received: number;
  bytes: number;
}

export interface SnapshotReceivedReply extends Envelope {
  result: SnapshotReceived;
}

export interface DeletedReply extends Envelope {
  result: Deleted;
}

export interface CodeReply extends Envelope {
  result: Code;
}

export interface ExperimentReply extends Envelope {
  result: Experiment;
}

export interface ExperimentCreatedReply extends Envelope {
  result: ExperimentCreated;
}

export interface ExperimentListReply extends Envelope {
  result: ExperimentList;
}

export interface ExperimentRunReply extends Envelope {
  result: ExperimentRun;
}

export interface ExperimentRunCreatedReply extends Envelope {
  result: ExperimentRunCreated;
}

export interface ExperimentRunListReply extends Envelope {
  result: ExperimentRunList;
}

export interface SliceListReply extends Envelope {
  result: SliceList;
}

export interface HeldRowsReply extends Envelope {
  result: HeldRows;
}

export interface DeployReply extends Envelope {
  result: Deploy;
}

export interface PredictorInfoReply extends Envelope {
  result: PredictorInfo;
}

export interface PredictorInfoListReply extends Envelope {
  result: PredictorInfoList;
}

export interface EstimateReply extends Envelope {
  result: Estimate;
}

export interface SourceReply extends Envelope {
  result: Source;
}

export interface SourceColumn {
  name: string;
  type: string;
  mode: "NULLABLE" | "REQUIRED" | "REPEATED";
}

/**
 * a table's or a query's first 20 rows. `table` or `sql` is what was read and
 * `location` where; `type` is the source's kind of table (`QUERY` for a query) and
 * `readable` whether its rows can be read and imported, `sample` empty when not. `rows`
 * and `bytes` are what the source says before reading, null where it says nothing.
 * `columns` name each column's type as the source names it; each sample cell is text,
 * cut to 256 characters, or null
 */
export interface SourcePreviewed {
  source: Name;
  table: string | null;
  sql: string | null;
  location: string;
  type: string;
  readable: boolean;
  rows: number | null;
  bytes: number | null;
  columns: SourceColumn[];
  sample: Array<Array<string | null>>;
}

export interface SourcePreviewedReply extends Envelope {
  result: SourcePreviewed;
}

export interface SourceTablesReply extends Envelope {
  result: SourceTables;
}

export interface KeyDeletedReply extends Envelope {
  result: KeyDeleted;
}

export interface ContextCreateRequest extends Envelope {
  payload: ContextCreate;
}

export interface ContextUploadRequest extends Envelope {
  payload: ContextUpload;
}

export interface ExperimentCreateRequest extends Envelope {
  payload: ExperimentCreate;
}

export interface ExperimentUpdateRequest extends Envelope {
  payload: ExperimentUpdate;
}

export interface ExperimentRunCreateRequest extends Envelope {
  payload: ExperimentRunCreate;
}

export interface PredictorUpdateRequest extends Envelope {
  payload: PredictorUpdate;
}

export interface PredictorDeployRequest extends Envelope {
  payload: PredictorDeploy;
}

export interface EstimateQueryRequest extends Envelope {
  payload: EstimateQuery;
}

export interface SourceCreateRequest extends Envelope {
  payload: SourceCreate;
}

export interface SourcePreviewRequest extends Envelope {
  payload: SourcePreview;
}

export const ROUTES = {
  "predictors.predict": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/predict", request: "QueriesRequest", reply: "ResultReply", body: "application/json" },
  "predict": { method: "POST", path: "/v1/projects/{project}/predict", request: "ManifestRequest", reply: "ResultReply", body: "application/json" },
  "predictions.get": { method: "GET", path: "/v1/projects/{project}/predictions/{id}", request: "null", reply: "ResultReply", body: null },
  "environments.list": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments", request: "null", reply: "EnvironmentListReply", body: null },
  "environments.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}", request: "null", reply: "EnvironmentReply", body: null },
  "environments.promote": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/promote", request: "PromoteRequest", reply: "EnvironmentReply", body: "application/json" },
  "environments.rollback": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/rollback", request: "PromoteRequest", reply: "EnvironmentReply", body: "application/json" },
  "environments.metrics": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/metrics", request: "null", reply: "MetricsReply", body: null },
  "environments.logs": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs", request: "null", reply: "LogsReply", body: null },
  "environments.logs.histogram": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs/histogram", request: "null", reply: "LogHistogramReply", body: null },
  "predictors.delete": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/delete", request: "EmptyRequest", reply: "PredictorDeletedReply", body: "application/json" },
  "contexts.list": { method: "GET", path: "/v1/projects/{project}/contexts", request: "null", reply: "ContextListReply", body: null },
  "contexts.create": { method: "POST", path: "/v1/projects/{project}/contexts", request: "ContextCreateRequest", reply: "ContextReply", body: "application/json" },
  "contexts.get": { method: "GET", path: "/v1/projects/{project}/contexts/{name}", request: "null", reply: "ContextReply", body: null },
  "contexts.versions": { method: "GET", path: "/v1/projects/{project}/contexts/{name}/versions", request: "null", reply: "ContextVersionsReply", body: null },
  "contexts.upload": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/versions", request: "ContextUploadRequest", reply: "ContextReply", body: "application/json" },
  "contexts.bytes": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/versions/{version}/bytes", request: "null", reply: "ReceivedReply", body: "application/octet-stream" },
  "contexts.refresh": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/refresh", request: "EmptyRequest", reply: "ContextReply", body: "application/json" },
  "contexts.append": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/append", request: "LiveContextAppendRequest", reply: "ReceiptReply", body: "application/json" },
  "contexts.label": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/label", request: "LiveContextLabelRequest", reply: "ReceiptReply", body: "application/json" },
  "contexts.rows": { method: "GET", path: "/v1/projects/{project}/contexts/{name}/rows", request: "null", reply: "RowsReply", body: null },
  "contexts.delete": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/delete", request: "EmptyRequest", reply: "DeletedReply", body: "application/json" },
  "contexts.code": { method: "GET", path: "/v1/projects/{project}/contexts/{name}/code", request: "null", reply: "CodeReply", body: null },
  "experiments.list": { method: "GET", path: "/v1/projects/{project}/experiments", request: "null", reply: "ExperimentListReply", body: null },
  "experiments.create": { method: "POST", path: "/v1/projects/{project}/experiments", request: "ExperimentCreateRequest", reply: "ExperimentCreatedReply", body: "application/json" },
  "experiments.get": { method: "GET", path: "/v1/projects/{project}/experiments/{name}", request: "null", reply: "ExperimentReply", body: null },
  "experiments.update": { method: "POST", path: "/v1/projects/{project}/experiments/{name}/update", request: "ExperimentUpdateRequest", reply: "ExperimentReply", body: "application/json" },
  "experiments.runs": { method: "GET", path: "/v1/projects/{project}/experiments/{name}/runs", request: "null", reply: "ExperimentRunListReply", body: null },
  "experiments.run": { method: "POST", path: "/v1/projects/{project}/experiments/{name}/runs", request: "ExperimentRunCreateRequest", reply: "ExperimentRunCreatedReply", body: "application/json" },
  "experiments.runs.get": { method: "GET", path: "/v1/projects/{project}/experiments/{name}/runs/{number}", request: "null", reply: "ExperimentRunReply", body: null },
  "experiments.cancel": { method: "POST", path: "/v1/projects/{project}/experiments/{name}/runs/{number}/cancel", request: "EmptyRequest", reply: "ExperimentRunReply", body: "application/json" },
  "experiments.rows": { method: "GET", path: "/v1/projects/{project}/experiments/{name}/runs/{number}/rows", request: "null", reply: "HeldRowsReply", body: null },
  "experiments.slices": { method: "GET", path: "/v1/projects/{project}/experiments/{name}/runs/{number}/slices", request: "null", reply: "SliceListReply", body: null },
  "experiments.delete": { method: "POST", path: "/v1/projects/{project}/experiments/{name}/delete", request: "EmptyRequest", reply: "DeletedReply", body: "application/json" },
  "experiments.code": { method: "GET", path: "/v1/projects/{project}/experiments/{name}/code", request: "null", reply: "CodeReply", body: null },
  "predictors.list": { method: "GET", path: "/v1/projects/{project}/predictors", request: "null", reply: "PredictorInfoListReply", body: null },
  "predictors.create": { method: "POST", path: "/v1/projects/{project}/predictors", request: "PredictorCreateRequest", reply: "DeployReply", body: "application/json" },
  "predictors.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}", request: "null", reply: "PredictorInfoReply", body: null },
  "predictors.update": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/update", request: "PredictorUpdateRequest", reply: "DeployReply", body: "application/json" },
  "predictors.deploy": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/deploy", request: "PredictorDeployRequest", reply: "DeployReply", body: "application/json" },
  "revisions.list": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/revisions", request: "null", reply: "RevisionListReply", body: null },
  "revisions.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/revisions/{number}", request: "null", reply: "RevisionReply", body: null },
  "predictors.code": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/code", request: "null", reply: "CodeReply", body: null },
  "estimate": { method: "POST", path: "/v1/projects/{project}/estimate", request: "EstimateQueryRequest", reply: "EstimateReply", body: "application/json" },
  "sources.list": { method: "GET", path: "/v1/projects/{project}/sources", request: "null", reply: "SourceListReply", body: null },
  "sources.create": { method: "POST", path: "/v1/projects/{project}/sources", request: "SourceCreateRequest", reply: "SourceReply", body: "application/json" },
  "sources.get": { method: "GET", path: "/v1/projects/{project}/sources/{source}", request: "null", reply: "SourceReply", body: null },
  "sources.tables": { method: "GET", path: "/v1/projects/{project}/sources/{source}/tables", request: "null", reply: "SourceTablesReply", body: null },
  "sources.preview": { method: "POST", path: "/v1/projects/{project}/sources/{source}/preview", request: "SourcePreviewRequest", reply: "SourcePreviewedReply", body: "application/json" },
  "sources.test": { method: "POST", path: "/v1/projects/{project}/sources/{source}/test", request: "EmptyRequest", reply: "SourceReply", body: "application/json" },
  "sources.delete": { method: "POST", path: "/v1/projects/{project}/sources/{source}/delete", request: "EmptyRequest", reply: "DeletedReply", body: "application/json" },
  "keys.delete": { method: "POST", path: "/v1/projects/{project}/keys/{key}/delete", request: "KeyRefRequest", reply: "KeyDeletedReply", body: "application/json" },
  "models.list": { method: "GET", path: "/v1/projects/{project}/models", request: "null", reply: "ModelListReply", body: null },
  "models.get": { method: "GET", path: "/v1/projects/{project}/models/{model}", request: "null", reply: "ModelReply", body: null },
  "ready": { method: "GET", path: "/readyz", request: "null", reply: "Health", body: null },
  "live": { method: "GET", path: "/livez", request: "null", reply: "Health", body: null },
  "projects.list": { method: "GET", path: "/v1/projects", request: "null", reply: "ProjectsReply", body: null },
  "projects.create": { method: "POST", path: "/v1/projects", request: "ProjectCreateRequest", reply: "ProjectReply", body: "application/json" },
  "keys.list": { method: "GET", path: "/v1/projects/{project}/keys", request: "null", reply: "KeyListReply", body: null },
  "keys.create": { method: "POST", path: "/v1/projects/{project}/keys", request: "KeyCreateRequest", reply: "KeyReply", body: "application/json" },
  "keys.revoke": { method: "POST", path: "/v1/projects/{project}/keys/{key}/revoke", request: "KeyRefRequest", reply: "KeyRevokedReply", body: "application/json" },
  "keys.rotate": { method: "POST", path: "/v1/projects/{project}/keys/{key}/rotate", request: "KeyRefRequest", reply: "KeyReply", body: "application/json" },
  "secrets.list": { method: "GET", path: "/v1/projects/{project}/secrets", request: "null", reply: "SecretListReply", body: null },
  "secrets.create": { method: "POST", path: "/v1/projects/{project}/secrets", request: "SecretCreateRequest", reply: "SecretReply", body: "application/json" },
  "secrets.get": { method: "GET", path: "/v1/projects/{project}/secrets/{secret}", request: "null", reply: "SecretReply", body: null },
  "secrets.rotate": { method: "POST", path: "/v1/projects/{project}/secrets/{secret}/rotate", request: "SecretRotateRequest", reply: "SecretReply", body: "application/json" },
  "secrets.delete": { method: "POST", path: "/v1/projects/{project}/secrets/{secret}/delete", request: "SecretRefRequest", reply: "SecretDeletedReply", body: "application/json" },
  "sources.prepare": { method: "POST", path: "/v1/projects/{project}/sources:prepare", request: "SourcePrepareRequest", reply: "SourcePreparedReply", body: "application/json" },
  "snapshots.list": { method: "GET", path: "/v1/projects/{project}/snapshots", request: "null", reply: "SnapshotListReply", body: null },
  "snapshots.create": { method: "POST", path: "/v1/projects/{project}/snapshots", request: "SnapshotCreateRequest", reply: "SnapshotReply", body: "application/json" },
  "snapshots.put": { method: "PUT", path: "/v1/projects/{project}/snapshots", request: "null", reply: "HeldSnapshotReply", body: "application/vnd.apache.arrow.stream" },
  "snapshots.create_synthetic": { method: "POST", path: "/v1/projects/{project}/snapshots/synthetic", request: "SyntheticBatchCreateRequest", reply: "SyntheticBatchCreatedReply", body: "application/json" },
  "snapshots.synthetic": { method: "GET", path: "/v1/projects/{project}/snapshots/synthetic/{name}", request: "null", reply: "SyntheticBatchReply", body: null },
  "snapshots.get": { method: "GET", path: "/v1/projects/{project}/snapshots/{snapshot}", request: "null", reply: "SnapshotReply", body: null },
  "snapshots.update": { method: "POST", path: "/v1/projects/{project}/snapshots/{snapshot}/update", request: "SnapshotUpdateRequest", reply: "SnapshotReply", body: "application/json" },
  "snapshots.delete": { method: "POST", path: "/v1/projects/{project}/snapshots/{snapshot}/delete", request: "EmptyRequest", reply: "SnapshotReply", body: "application/json" },
  "snapshots.bytes": { method: "POST", path: "/v1/projects/{project}/snapshots/{snapshot}/bytes", request: "null", reply: "SnapshotReceivedReply", body: "application/octet-stream" },
  "runs.list": { method: "GET", path: "/v1/projects/{project}/runs", request: "null", reply: "RunListReply", body: null },
  "runs.create": { method: "POST", path: "/v1/projects/{project}/runs", request: "RunCreateRequest", reply: "RunReply", body: "application/json" },
  "runs.get": { method: "GET", path: "/v1/projects/{project}/runs/{run}", request: "null", reply: "RunReply", body: null },
  "runs.cancel": { method: "POST", path: "/v1/projects/{project}/runs/{run}/cancel", request: "RunCancelRequest", reply: "RunReply", body: "application/json" },
  "runs.lineage": { method: "GET", path: "/v1/projects/{project}/runs/{run}/lineage", request: "null", reply: "RunLineageReply", body: null },
  "auth.status": { method: "GET", path: "/v1/auth/status", request: "null", reply: "AuthStatusReply", body: null },
  "auth.logout": { method: "POST", path: "/v1/auth/logout", request: "EmptyRequest", reply: "LoggedOutReply", body: "application/json" },
  "workspace.get": { method: "GET", path: "/v1/workspace", request: "null", reply: "ProjectListReply", body: null },
  "organizations.list": { method: "GET", path: "/v1/organizations", request: "null", reply: "OrganizationListReply", body: null },
  "organizations.create": { method: "POST", path: "/v1/organizations", request: "OrganizationCreateRequest", reply: "OrganizationReply", body: "application/json" },
  "sso.get": { method: "GET", path: "/v1/projects/{project}/sso", request: "null", reply: "SsoReply", body: null },
  "sso.configure": { method: "POST", path: "/v1/projects/{project}/sso", request: "SsoConfigureRequest", reply: "SsoReply", body: "application/json" },
  "audit.list": { method: "GET", path: "/v1/projects/{project}/audit", request: "null", reply: "AuditLogReply", body: null },
  "device.start": { method: "POST", path: "/v1/device-logins", request: "DeviceLoginStart", reply: "DeviceLoginGrant", body: "application/json" },
  "device.token": { method: "POST", path: "/v1/device-logins/token", request: "DeviceLoginPoll", reply: "DeviceLoginToken", body: "application/json" },
  "onboarding.get": { method: "GET", path: "/v1/onboarding", request: "null", reply: "OnboardingReply", body: null },
  "workspaces.list": { method: "GET", path: "/v1/workspaces", request: "null", reply: "WorkspaceListReply", body: null },
  "workspaces.create": { method: "POST", path: "/v1/workspaces", request: "WorkspaceCreateRequest", reply: "WorkspaceCreatedReply", body: "application/json" },
  "workspaces.get": { method: "GET", path: "/v1/workspaces/{workspace}", request: "null", reply: "WorkspaceDetailsReply", body: null },
  "workspaces.update": { method: "POST", path: "/v1/workspaces/{workspace}", request: "WorkspaceRenameRequest", reply: "WorkspaceSummaryReply", body: "application/json" },
  "workspaces.join": { method: "POST", path: "/v1/workspaces/{workspace}/join", request: "WorkspaceRefRequest", reply: "WorkspaceJoinReply", body: "application/json" },
  "workspaces.members": { method: "GET", path: "/v1/workspaces/{workspace}/members", request: "null", reply: "WorkspaceMemberListReply", body: null },
  "members.update": { method: "POST", path: "/v1/workspaces/{workspace}/members/role", request: "MemberRoleChangeRequest", reply: "MemberRoleChangedReply", body: "application/json" },
  "members.remove": { method: "POST", path: "/v1/workspaces/{workspace}/members/remove", request: "MemberRemoveRequest", reply: "MemberRemovedReply", body: "application/json" },
  "invitations.list": { method: "GET", path: "/v1/workspaces/{workspace}/invitations", request: "null", reply: "InvitationListReply", body: null },
  "invitations.create": { method: "POST", path: "/v1/workspaces/{workspace}/invitations", request: "InvitationCreateRequest", reply: "InvitationCreatedReply", body: "application/json" },
  "invitations.revoke": { method: "POST", path: "/v1/workspaces/{workspace}/invitations/{id}/revoke", request: "InvitationRevokeRequest", reply: "InvitationViewReply", body: "application/json" },
  "requests.list": { method: "GET", path: "/v1/workspaces/{workspace}/join-requests", request: "null", reply: "JoinRequestListReply", body: null },
  "requests.approve": { method: "POST", path: "/v1/workspaces/{workspace}/join-requests/{id}/approve", request: "JoinRequestApproveRequest", reply: "JoinRequestApprovedReply", body: "application/json" },
  "requests.deny": { method: "POST", path: "/v1/workspaces/{workspace}/join-requests/{id}/deny", request: "JoinRequestDenyRequest", reply: "JoinRequestDeniedReply", body: "application/json" },
  "domains.list": { method: "GET", path: "/v1/workspaces/{workspace}/domains", request: "null", reply: "JoinDomainListReply", body: null },
  "domains.configure": { method: "POST", path: "/v1/workspaces/{workspace}/domains", request: "JoinDomainConfigureRequest", reply: "JoinDomainConfiguredReply", body: "application/json" },
  "invitations.preview": { method: "POST", path: "/v1/invitations/preview", request: "InvitationTokenRequest", reply: "InvitationPreviewReply", body: "application/json" },
  "invitations.accept": { method: "POST", path: "/v1/invitations/accept", request: "InvitationTokenRequest", reply: "InvitationAcceptedReply", body: "application/json" },
  "requests.cancel": { method: "POST", path: "/v1/join-requests/{id}/cancel", request: "JoinRequestRefRequest", reply: "JoinRequestViewReply", body: "application/json" },
  "waitlist.join": { method: "POST", path: "/v1/waitlist", request: "WaitlistJoinRequest", reply: "WaitlistStatusReply", body: "application/json" },
  "waitlist.get": { method: "GET", path: "/v1/waitlist/{id}", request: "null", reply: "WaitlistEntryViewReply", body: null },
  "waitlist.approve": { method: "POST", path: "/v1/waitlist/{id}/approve", request: "WaitlistRefRequest", reply: "WaitlistEntryViewReply", body: "application/json" },
  "device.preview": { method: "POST", path: "/v1/device-logins/preview", request: "DeviceLoginCodeRequest", reply: "DeviceLoginPreviewReply", body: "application/json" },
  "device.approve": { method: "POST", path: "/v1/device-logins/approve", request: "DeviceLoginApproveRequest", reply: "DeviceLoginApprovedReply", body: "application/json" },
  "device.deny": { method: "POST", path: "/v1/device-logins/deny", request: "DeviceLoginCodeRequest", reply: "DeviceLoginDeniedReply", body: "application/json" },
} as const;

export type RouteName = keyof typeof ROUTES;
