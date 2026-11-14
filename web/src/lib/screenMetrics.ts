import { useJson } from './data';
import type { SurvivalModel } from './screen';

/** Shapes of web/public/screen/*.json, written by ml/qihgnn/screen/train.py. Every field is optional:
 *  until the models are trained the site shows that state rather than invented numbers. */
export interface Binary {
  n: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  sensitivity: number;
  sensitivity_ci: [number, number];
  specificity: number;
  specificity_ci: [number, number];
  accuracy: number;
  accuracy_ci: [number, number];
  auc: number | null;
}
export interface Summary {
  n: number;
  mean: number | null;
  sd?: number | null;
  median?: number | null;
  q25?: number | null;
  q75?: number | null;
}

export interface ScreenMetrics {
  generated_at: string;
  device: string;
  probe: boolean;
  minutes?: number;
  brats?: { volumes_found: number; volumes_used: number; splits: Record<string, number>; id_offset: number };
  dedup_audit?: {
    images: number;
    duplicate_clusters: number;
    images_in_duplicate_clusters: number;
    official_test_images: number;
    official_test_images_with_a_copy_in_training: number;
    clusters_with_conflicting_labels: number;
  };
  classifier_data?: Record<'train' | 'val' | 'test', { n: number; per_class: number[] }>;
  classifier?: {
    architecture: string;
    temperature: number;
    history: { epoch: number; train_loss: number | null; val_macro_f1: number; val_acc: number }[];
    test: {
      n: number;
      accuracy: number;
      accuracy_ci: [number, number];
      macro_f1: number;
      per_class: Record<string, { n: number; precision: number; recall: number; recall_ci: [number, number] }>;
      confusion: number[][];
      labels: string[];
      tumour_vs_no_tumour: Binary;
      not_brain_mri_detection: Binary;
      by_source: Record<string, { n: number; accuracy: number }>;
      ece: number;
      ece_before_calibration?: number;
      abstention: { threshold: number; coverage: number; accuracy_when_answered: number | null };
    };
    external_test?: Binary & { rejected_as_not_brain_mri: number; removed_as_copies_of_training_data: number };
  };
  segmenter?: {
    architecture: string;
    history: { epoch: number; train_loss: number | null; val_dice_tumour_slices: number }[];
    brats_slices: Record<string, { dice_on_tumour_slices: Summary; false_alarm_rate_on_tumour_free_slices: number | null; tumour_free_slices: number }>;
    figshare_test?: { dice: Summary; note: string };
    brats_3d_test: Record<string, {
      patients: number;
      dice3d: Summary;
      volume_abs_error_cc: Summary;
      volume_rel_error: Summary;
      volume_pearson_r: number | null;
      pairs: [number, number][];
    }>;
  };
  figshare?: { pairs: number; splits: Record<string, number> };
  survival_matching?: { survival_rows: number; matched_volumes: number };
  survival?: {
    features: string[];
    n: number;
    events: number;
    cv_c_index: Record<string, { mean: number; sd: number; folds: number }>;
    risk_groups: { km: Record<'low' | 'intermediate' | 'high', { t: number[]; s: number[]; n: number }>; logrank_p: number };
    survival_class_accuracy: { accuracy: number; majority_class_baseline: number; n: number; classes_days: number[] };
    hazard_ratio_per_sd: number[];
    ages?: Summary;
    wt_volume_cc?: Summary;
  };
  export?: Record<'classifier' | 'segmenter', { file: string; bytes: number; parameters?: number; max_abs_diff_vs_torch?: number }>;
}

export interface Sample {
  file: string;
  truth: string;
  source: string;
  kind?: 'volume';
  true_wt_cc?: number;
}

export function useScreen() {
  const metrics = useJson<ScreenMetrics>('screen/screen_metrics.json');
  const survival = useJson<SurvivalModel>('screen/survival.json');
  const samples = useJson<Sample[]>('screen/samples/index.json');
  const ready = !!metrics.data && !metrics.data.probe && !!metrics.data.export;
  return { metrics: metrics.data, survival: survival.data, samples: samples.data ?? [], loading: metrics.loading, ready };
}

export const pct = (x: number | null | undefined, digits = 1) => (x === null || x === undefined || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(digits)}%`);
export const ci = (c: [number, number] | undefined) => (c ? `${(c[0] * 100).toFixed(1)}–${(c[1] * 100).toFixed(1)}%` : '');
