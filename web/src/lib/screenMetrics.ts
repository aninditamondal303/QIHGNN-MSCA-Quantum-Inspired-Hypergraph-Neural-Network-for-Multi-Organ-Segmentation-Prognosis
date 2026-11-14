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
    external_test?: Binary & { rejected_as_not_brain_mri: number; removed_as_copies_of_training_data: number; internal_duplicates_removed?: number };
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

export interface OrganMetrics {
  generated_at: string;
  probe: boolean;
  liver_data?: { source?: string; volumes: number; slices: number | null; splits: Record<string, number> };
  liver?: {
    architecture: string;
    stage2?: { input: number; parameters: number };
    label_cleaning?: string;
    tumour_threshold: number;
    slice_rule_tumour_px: number;
    test_patients: number;
    patients_with_tumour: number;
    volume_dice_liver: Summary;
    volume_dice_tumour: Summary;
    slice_detection: (Binary & { auc?: number }) | null;
    lesions: { n: number; found: number; recall: number; recall_large: number; n_large: number; recall_small: number; n_small: number; false_per_patient: Summary };
    burden_pearson_r: number | null;
    burden_pairs: [number, number][];
    lesion_count_pairs: [number, number][];
    liver_volume_error_pct?: Summary;
    liver_volume_r?: number | null;
    tumour_volume_error_cc?: Summary;
    tumour_volume_r?: number | null;
    note: string;
  };
  cervix_data?: { images: number; slide_groups: number; splits: Record<string, number>; cells_annotated: number; nuclei_annotated: number };
  cervix?: {
    architecture: string;
    parameters: number;
    test_images: number;
    dice_nucleus: Summary;
    dice_cell: Summary;
    nucleus_count_abs_error: Summary;
    nucleus_count_pearson_r: number | null;
    nucleus_count_pairs: [number, number][];
    median_nc_abs_error: Summary;
    median_nc_pearson_r: number | null;
    training_nc_distribution: Summary;
    unusual_nc_threshold: number | null;
    note: string;
  };
  export?: Partial<Record<'liver' | 'liver_tumour' | 'cervix', { file: string; bytes: number; max_abs_diff_vs_torch?: number }>>;
}

export interface OrganSample {
  file: string;
  truth: string;
  kind: 'liver' | 'liver-volume' | 'cervix';
  true_tumour_px?: number;
  true_liver_px?: number;
  true_lesions?: number;
  source: string;
  nuclei_annotated?: number;
}

export function useOrgans() {
  const metrics = useJson<OrganMetrics>('screen/organs_metrics.json');
  const samples = useJson<OrganSample[]>('screen/samples/organs_index.json');
  const ready = !!metrics.data && !metrics.data.probe && !!metrics.data.export;
  // Each organ goes live on its own: a model is published only with its measured accuracy.
  const liverReady = ready && !!metrics.data?.liver?.tumour_threshold && !!metrics.data?.export?.liver && !!metrics.data?.export?.liver_tumour;
  const cervixReady = ready && !!metrics.data?.cervix && !!metrics.data?.export?.cervix;
  return { metrics: metrics.data, samples: samples.data ?? [], loading: metrics.loading, ready, liverReady, cervixReady };
}

/** The 3D volume evaluation for the sequence that performed best (T2 in the published run). */
export function best3d(m: ScreenMetrics | null | undefined) {
  const e = Object.entries(m?.segmenter?.brats_3d_test ?? {});
  e.sort((a, b) => (b[1].dice3d.mean ?? 0) - (a[1].dice3d.mean ?? 0));
  return e[0] ? { sequence: e[0][0], ...e[0][1] } : undefined;
}

export const SOURCE_NAMES: Record<string, string> = {
  'brain-mri-4class': 'Brain MRI dataset',
  'brats-slices': 'BraTS slices',
  'abdominal-ct': 'Abdominal CT',
  'colposcopy-photo': 'Cervical cell images',
  'cervical-cytology': 'Cervical cell images',
  synthetic: 'Synthetic non-medical',
};
