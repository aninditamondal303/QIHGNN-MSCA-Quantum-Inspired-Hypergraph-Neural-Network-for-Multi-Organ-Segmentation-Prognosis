export interface Stage {
  n: number;
  key: string;
  name: string;
  short: string;
  budgetKey: string[] | null; // keys into the parameter budget
  shapeIn: string;
  shapeOut: string;
  proposed: boolean;
  body: string;
  tex: string | null;
  note?: string;
}

export const STAGES: Stage[] = [
  {
    n: 1,
    key: 'input',
    name: 'Input volume',
    short: 'Input',
    budgetKey: null,
    shapeIn: '—',
    shapeOut: 'B × 4 × 128 × 128 × 64',
    proposed: false,
    body: 'Four channels per volume: the four MRI sequences (T1, T1ce, T2, FLAIR) for brain studies, or four Hounsfield windows of the same CT scan for abdominal studies. Every sample also carries a supervision vector recording which organs its source dataset labels.',
    tex: 'X \\in \\mathbb{R}^{B\\times 4\\times H\\times W\\times D},\\quad m \\in \\{0,1\\}^{K+1}',
  },
  {
    n: 2,
    key: 'encoder',
    name: '3D convolutional encoder',
    short: 'Encoder',
    budgetKey: ['encoder'],
    shapeIn: 'B × 4 × 128 × 128 × 64',
    shapeOut: 'B × 256 × 16 × 16 × 8',
    proposed: false,
    body: 'A residual stem and three strided residual blocks double the channels from 32 to 256 while halving resolution. Instance normalisation replaces batch normalisation because two volumes per batch give unreliable batch statistics. Intermediate maps are kept as skip connections for the decoder.',
    tex: 'h_{\\ell+1} = \\sigma\\big(\\mathrm{IN}(W_2 * \\sigma(\\mathrm{IN}(W_1 * h_\\ell))) + P\\,h_\\ell\\big)',
  },
  {
    n: 3,
    key: 'mpo',
    name: 'Matrix product operator',
    short: 'MPO',
    budgetKey: ['mpo'],
    shapeIn: 'B × 256 × 16 × 16 × 8',
    shapeOut: 'B × 256 × 16 × 16 × 8',
    proposed: true,
    body: 'The 256 → 256 channel map is factorised as a chain of two small tensors, the construction many-body physics calls a matrix product operator. At bond dimension 32 it holds 16,384 weights instead of 65,536. The stage is 0.21% of the network, so its job is to make channel mixing cheap, not to add capacity.',
    tex: 'y_{\\beta_1\\beta_2} = \\sum_{\\alpha_1\\alpha_2} G_1[\\alpha_1,\\beta_1,r]\\,G_2[r,\\alpha_2,\\beta_2]\\;x_{\\alpha_1\\alpha_2} + b',
  },
  {
    n: 4,
    key: 'hypergraph',
    name: 'Hypergraph convolution',
    short: 'Hypergraph',
    budgetKey: ['hypergraph'],
    shapeIn: 'B × 256 × 16 × 16 × 8',
    shapeOut: 'B × 256 × 16 × 16 × 8',
    proposed: true,
    body: 'The feature map is pooled to 256 region nodes. A learned projection gives each node a soft membership in five organ hyperedges, so a statement like "these seven regions are liver" is one relation instead of 21 pairwise edges. Information moves node → organ → node, letting distant parts of one organ exchange evidence in a single step.',
    tex: "Z' = \\phi\\big(D_v^{-1/2} H W_e D_e^{-1} H^{\\top} D_v^{-1/2} Z\\Theta + Z\\big),\\quad H = \\sigma(\\mathrm{MLP}(Z))",
  },
  {
    n: 5,
    key: 'msca',
    name: 'Multi-scale cross-attention',
    short: 'MSCA',
    budgetKey: ['msca'],
    shapeIn: 'B × 256 × 16 × 16 × 8',
    shapeOut: 'B × 256 × 16 × 16 × 8',
    proposed: true,
    body: 'Attention stays global but runs over four fixed token budgets (256, 32, 4 and 1) instead of every voxel. Each finer scale then attends to the next coarser one. The cost is 66,577 attention entries per head, compared with 1.1 × 10¹² for voxel attention, and it does not grow with input size.',
    tex: '\\Omega = \\sum_s n_s^2 = 256^2 + 32^2 + 4^2 + 1^2 = 66{,}577,\\qquad \\partial\\Omega/\\partial V = 0',
  },
  {
    n: 6,
    key: 'shared',
    name: 'Shared representation',
    short: 'Shared',
    budgetKey: ['shared'],
    shapeIn: 'B × 256 × 16 × 16 × 8',
    shapeOut: 'B × 512',
    proposed: false,
    body: 'Global pooling, a linear map and layer normalisation produce one 512-dimensional encoding of the patient. Only after this point does the network split, so segmentation and prognosis read the same anatomy instead of two pipelines joined by a mask.',
    tex: 'h = \\mathrm{LN}\\big(W\\,\\mathrm{GAP}(F) + b\\big) \\in \\mathbb{R}^{512}',
  },
  {
    n: 7,
    key: 'heads',
    name: 'Segmentation and prognosis heads',
    short: 'Heads',
    budgetKey: ['segmentation_head', 'prognosis_head'],
    shapeIn: 'B × 512 (+ skips, + 5 covariates)',
    shapeOut: 'B × 6 × 128 × 128 × 64  ·  B × 5',
    proposed: false,
    body: 'A transposed-convolution decoder with skip connections returns six channels (five organs and background). The shared vector conditions it through a parameter-free scale and shift. The prognosis head combines the shared vector with five covariates and returns a log-risk, 1-, 3- and 5-year survival logits, and an uncertainty term.',
    tex: '\\mathcal{L} = \\sum_{t} e^{-s_t}\\mathcal{L}_t + s_t,\\quad t \\in \\{\\text{mar},\\text{exc},\\text{cox},\\text{hor}\\}',
  },
  {
    n: 8,
    key: 'output',
    name: 'Decision-support output',
    short: 'Output',
    budgetKey: null,
    shapeIn: 'B × 6 × 128 × 128 × 64  ·  B × 5',
    shapeOut: 'mask · risk · 1/3/5-year survival',
    proposed: false,
    body: 'Before anything is returned, the serving layer reads from the checkpoint which channels never received a gradient and sets them to zero. The deployed system therefore cannot report an organ it was never trained on.',
    tex: '\\hat{y}_v = \\arg\\max_{c} \\; p_{c,v}\\,\\mathbb{1}[c \\in \\mathcal{S}_{\\text{trained}}]',
  },
];
