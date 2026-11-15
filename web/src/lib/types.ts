export type Label = 'background' | 'brain' | 'liver' | 'kidney' | 'cervix' | 'spleen';

export interface CaseChannel {
  name: string;
  file: string;
}

export interface CaseInfo {
  id: string;
  title: string;
  modality: 'MRI' | 'CT';
  analogOf: string;
  synthetic: boolean;
  shape: [number, number, number];
  spacing?: [number, number, number];
  mosaicColumns: number;
  channels: CaseChannel[];
  label: string;
  supervised: Label[];
  present: Label[];
  voxels: Record<string, number>;
  note: string;
  /** Real held-out cases: per-channel [lo, hi] that maps the 8-bit PNG back to training values. */
  channelRanges?: [number, number][];
  covariates?: number[];
  /** Dice of the in-browser network against the expert on this case, measured when it was published. */
  networkDice?: Record<string, number | null>;
  model?: { run_id: string | null; suppressed: string[] };
}

export interface CasesFile {
  labels: Label[];
  labelDescriptions: Record<Label, string>;
  cases: CaseInfo[];
}

export interface Check {
  passed: boolean;
}

export interface Verification {
  generated_at: string;
  git: string | null;
  environment: { python: string; numpy: string; platform: string };
  parameters: {
    stages: Record<string, number>;
    total: number;
    fp32_megabytes: number;
    report_stages: Record<string, number>;
    report_total: number;
    matches_report: boolean;
  };
  mpo_compression: { bond: number; parameters: number; dense: number; compression: number }[];
  attention: { token_budgets: number[]; at_resolution: { shape: number[]; voxels: number; multi_scale: number; naive: number; reduction: number }[] };
  marginal_loss: Check & {
    seed: number;
    voxels: number;
    supervised_channels: string[];
    rows: {
      channel: string;
      marginal_original: number;
      marginal_perturbed: number;
      marginal_change: number;
      naive_original: number;
      naive_perturbed: number;
      naive_change: number;
    }[];
  };
  unlabelled_prediction: Check & {
    fraction_moved: number;
    rows: { channel: string; marginal_change: number; naive_change: number }[];
    max_marginal_change: number;
    min_naive_increase: number;
  };
  cox: Check & { loss_original: number; loss_shifted_37_5: number; loss_all_censored: number; cases: number; events: number };
  uncertainty_weighting: Check & { task_losses: number[]; converged_log_vars: number[]; max_abs_error: number };
  dice: Check & { cases: Record<string, { observed: number; expected: number }> };
  horizon_labels: Check;
  exclusion: Check & { softmax_head: number; sigmoid_head: number; finding: string };
  concordance: Check;
  checks: Record<string, boolean>;
  all_passed: boolean;
}

export interface CorpusEntry {
  key: string;
  name: string;
  kaggle: string | null;
  modality: string;
  claimed_channels: Label[];
  expected_format: string;
  report_cases: number | null;
  notes: string;
  status: 'admitted' | 'rejected' | 'pending inspection' | 'not registered';
  supervises: Label[];
  reason: string;
  cases: number | null;
}

export interface CorpusFile {
  inspected: boolean;
  inspectedAt: string | null;
  labels: Label[];
  corpora: CorpusEntry[];
  supervisedChannels: Label[];
  unsupervisedChannels: Label[];
  reportCorpus: { cases: number; rows: { name: string; modality: string; format: string; cases: number | null; supervises: string }[] };
}

export interface MetricsRun {
  label: string;
  run_id: string | null;
  epoch: number | null;
  evaluated_on: string;
  cases: number;
  suppressed_channels: string[];
  segmentation: Record<string, { cases: number; dice_mean: number; dice_sd: number; hd95_mean: number | null }>;
  prognosis: null | { cases: number; events: number; c_index: number | null; td_auc: Record<string, number | null> };
}

export interface MetricsFile {
  generated_at: string;
  runs: MetricsRun[];
}

export interface TrainingHistory {
  run_id: string;
  device: string;
  train_cases: number;
  val_cases: number;
  train_config: { dataset: string; epochs: number };
  history: { epoch: number; train_total: number | null; train_marginal: number | null; val_marginal: number | null }[];
}

export interface Health {
  status: string;
  version: string;
  torch_available: boolean;
  checkpoint: string | null;
  inference_ready: boolean;
}

export interface PredictResponse {
  case_id: string;
  run_id: string | null;
  shape: number[];
  mosaic_columns: number;
  label_png_base64: string;
  suppressed: string[];
  log_risk: number;
  survival_probability: Record<string, number>;
  aleatoric_sd: number;
  seconds: number;
}
