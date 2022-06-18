
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 18:21:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03326E+00  1.00029E+00  1.00473E+00  9.99249E-01  1.00021E+00  9.83393E-01  9.99177E-01  1.00964E+00  9.98367E-01  9.89708E-01  9.82165E-01  9.98687E-01  9.87070E-01  9.92006E-01  9.96565E-01  1.00862E+00  1.00387E+00  1.01300E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.58987E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74101E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39640E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53523E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60055E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.43788E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.42862E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.09857E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.14717E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500521 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50036E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50036E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.75999E+02 ;
RUNNING_TIME              (idx, 1)        =  1.92248E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.02367E-01  1.02367E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.81344E+01  1.81344E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.11550E-01  7.60383E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.85126E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.15477 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97778E+00 2.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.08720E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  6.00000E-04 ;
URES_EMAX                 (idx, 1)        =  9.99999E-02 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 3 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  2.84851E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.24012E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.12901E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.84851E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.24012E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.73771E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.45330E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.73771E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.45330E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29315E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.12804E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.30093E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.40165E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20796E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.53799E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.52516E+18 0.00373  1.62575E-02 0.00366 ];
U233_FISS                 (idx, [1:   4]) = [  9.22815E+19 0.00047  9.83743E-01 6.0E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.02662E+19 0.00055  7.54075E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34407E+19 0.00131  1.12285E-01 0.00127 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001457 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16085E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001457 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5087617 5.09292E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3987105 3.99113E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 926735 9.27558E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001457 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.34693E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36462E+20 3.5E-06  2.36462E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.2E-07  9.37764E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.19645E+20 0.00031  1.06690E+20 0.00024  1.29558E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.13422E+20 0.00017  2.00466E+20 0.00013  1.29558E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.35041E+20 0.00036  2.35041E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.09223E+22 0.00026  2.93476E+22 0.00024  1.57461E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.18026E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.35224E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.04205E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51105E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.20046E-01 0.07868 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.76395E-01 0.00852 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.15633E-04 0.01417 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.44580E+03 0.02597 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07255E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99988E-01 1.2E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.71527E-01 0.07054 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.37265E-01 0.07054 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52155E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.00630E+00 0.00042  6.26951E-02 0.00041  2.04063E-04 0.00883 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.00644E+00 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  1.00610E+00 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.00644E+00 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.10938E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02170E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02399E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33893E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33565E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48904E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48621E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.21022E-03 0.00572  2.64276E-04 0.01821  7.81449E-04 0.01165  5.93640E-04 0.01272  1.25914E-03 0.00930  2.63095E-04 0.01981  4.86217E-05 0.04510 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.44371E-01 0.01452  1.24756E-02 3.2E-05  3.23390E-02 9.5E-05  1.05975E-01 0.00060  2.97199E-01 0.00025  1.23576E+00 0.00023  4.68523E+00 0.04133 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22403E-03 0.00796  2.66222E-04 0.02790  7.79053E-04 0.01778  5.95199E-04 0.01818  1.27171E-03 0.01329  2.64099E-04 0.02977  4.77476E-05 0.06832 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46590E-01 0.02346  1.24761E-02 3.6E-05  3.23374E-02 0.00014  1.05935E-01 0.00078  2.97155E-01 0.00037  1.23592E+00 0.00030  6.51572E+00 0.02865 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.16011E-07 0.00125  8.15660E-07 0.00125  9.20299E-07 0.01716 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.21102E-07 0.00120  8.20749E-07 0.00121  9.25892E-07 0.01702 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.25029E-03 0.00926  2.69373E-04 0.03112  7.79694E-04 0.01824  6.05248E-04 0.02122  1.28143E-03 0.01477  2.67428E-04 0.03080  4.71124E-05 0.07022 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.45718E-01 0.02347  1.24756E-02 5.0E-05  3.23441E-02 0.00017  1.05906E-01 0.00095  2.97152E-01 0.00043  1.23609E+00 0.00037  6.68165E+00 0.03932 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.24487E-07 0.01706  7.24049E-07 0.01706  8.32974E-07 0.05389 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.29068E-07 0.01705  7.28627E-07 0.01705  8.38429E-07 0.05384 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83736E-03 0.04214  3.05796E-04 0.12655  6.70435E-04 0.07767  5.58973E-04 0.08180  1.01986E-03 0.06618  2.49530E-04 0.12122  3.27704E-05 0.41892 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.97261E-01 0.08013  1.24770E-02 0.00011  3.23287E-02 0.00050  1.05491E-01 0.00235  2.97391E-01 0.00155  1.23789E+00 0.00087  5.45663E+00 0.15206 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.83108E-03 0.04157  2.98543E-04 0.12582  6.65928E-04 0.07705  5.54974E-04 0.08137  1.03458E-03 0.06550  2.44408E-04 0.12033  3.26411E-05 0.40094 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.99091E-01 0.07964  1.24770E-02 0.00011  3.23278E-02 0.00050  1.05520E-01 0.00237  2.97421E-01 0.00154  1.23793E+00 0.00086  5.45663E+00 0.15206 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96736E+03 0.03952 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.07894E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12923E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22861E-03 0.00578 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.99785E+03 0.00590 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.24366E-09 0.00133 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.20397E-05 0.00424  7.20514E-05 0.00425  1.64396E-06 0.32714 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.86338E-05 0.01513  7.86369E-05 0.01518  1.79107E-06 0.37909 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.97825E-04 0.01342  4.98423E-04 0.01339  3.12326E-04 0.31330 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73309E+01 0.01120 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.42862E+01 0.00027  4.01527E+01 0.00033 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 18:51:21 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00915E+00  9.99416E-01  9.96718E-01  9.93805E-01  9.90741E-01  1.00050E+00  9.92760E-01  1.00755E+00  9.89624E-01  9.99161E-01  9.99533E-01  9.99729E-01  1.00286E+00  1.00546E+00  1.00170E+00  1.00001E+00  1.00725E+00  1.00402E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 6.6E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.58485E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74152E-01 9.0E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39501E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53364E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59979E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.44227E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.43298E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.11217E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.13909E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500454 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50055E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50055E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.49885E+02 ;
RUNNING_TIME              (idx, 1)        =  4.88645E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.83733E-01  8.13667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.76698E+01  2.95354E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.64333E-02  1.64333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.22605E+00  2.41445E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.64992E+01  2.83517E+02 ];
CPU_USAGE                 (idx, 1)        = 9.20680 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98413E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.13918E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 172 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  5.73973E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60121E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.64732E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.44424E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.01710E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26508E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46899E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.41788E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.64565E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.35461E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.67170E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06326E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17848E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44270E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35725E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17620E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55516E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60446E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38180E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.37444E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.50926E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51831E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.23098E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.48956E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.07980E-04 -2.38416E+25  2.20820E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.64399E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.53892E+18 0.00436  1.64093E-02 0.00426 ];
U233_FISS                 (idx, [1:   4]) = [  9.22074E+19 0.00051  9.83316E-01 7.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  5.46124E+14 0.20270  5.81047E-06 0.20272 ];
TH232_CAPT                (idx, [1:   4]) = [  9.12396E+19 0.00051  7.49959E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34116E+19 0.00141  1.10237E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  9.55295E+13 0.49819  7.81894E-07 0.49813 ];
SM149_CAPT                (idx, [1:   4]) = [  1.24277E+16 0.04218  1.02173E-04 0.04218 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002184 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16425E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002184 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5123126 5.12820E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3949049 3.95267E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 930009 9.30765E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002184 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.52737E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36455E+20 3.6E-06  2.36455E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37761E+19 1.2E-07  9.37761E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21596E+20 0.00031  1.08571E+20 0.00023  1.30245E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15372E+20 0.00017  2.02348E+20 0.00012  1.30245E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.37239E+20 0.00037  2.37239E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.12390E+22 0.00027  2.96533E+22 0.00025  1.58566E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20823E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.37454E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05282E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51195E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51195E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.71329E-01 0.07123 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.85733E-01 0.00815 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.40653E-04 0.01522 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.54679E+03 0.02188 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06930E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.45019E-01 0.06068 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.03630E-01 0.06068 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52148E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.96822E-01 0.00042  6.20900E-02 0.00041  1.98957E-04 0.00894 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.96960E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.96748E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.96960E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09930E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02649E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02661E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33253E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33216E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48529E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48989E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.22416E-03 0.00581  2.58859E-04 0.02012  7.90286E-04 0.01146  5.90883E-04 0.01234  1.27083E-03 0.00928  2.64173E-04 0.02019  4.91299E-05 0.04731 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.41012E-01 0.01562  1.24751E-02 3.4E-05  3.23361E-02 8.8E-05  1.05945E-01 0.00054  2.97300E-01 0.00028  1.23238E+00 0.00252  4.32708E+00 0.04350 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21056E-03 0.00828  2.62981E-04 0.02969  7.88614E-04 0.01742  5.78571E-04 0.01864  1.26714E-03 0.01311  2.61418E-04 0.02775  5.18402E-05 0.06867 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45513E-01 0.02339  1.24750E-02 4.3E-05  3.23345E-02 0.00013  1.06024E-01 0.00082  2.97246E-01 0.00037  1.23552E+00 0.00031  6.04692E+00 0.03030 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.25261E-07 0.00121  8.24956E-07 0.00121  9.09610E-07 0.01550 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.22576E-07 0.00112  8.22273E-07 0.00112  9.06597E-07 0.01548 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20655E-03 0.00893  2.61583E-04 0.03236  7.82765E-04 0.01747  5.96738E-04 0.02066  1.25145E-03 0.01342  2.69559E-04 0.03058  4.44559E-05 0.07290 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.39248E-01 0.02317  1.24748E-02 5.0E-05  3.23312E-02 0.00016  1.05991E-01 0.00096  2.97247E-01 0.00046  1.23581E+00 0.00038  6.45316E+00 0.04197 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.34359E-07 0.01720  7.34247E-07 0.01720  6.80520E-07 0.05027 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.32236E-07 0.01720  7.32126E-07 0.01720  6.78410E-07 0.05031 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83579E-03 0.04242  2.59723E-04 0.13523  6.67832E-04 0.08079  5.01967E-04 0.09468  1.10929E-03 0.06639  2.42657E-04 0.11693  5.43235E-05 0.25862 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.81363E-01 0.09028  1.24746E-02 0.00015  3.23457E-02 0.00054  1.06606E-01 0.00366  2.97751E-01 0.00160  1.23684E+00 0.00100  6.43066E+00 0.10759 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.80107E-03 0.04161  2.56289E-04 0.13605  6.53296E-04 0.08017  4.92593E-04 0.09396  1.10080E-03 0.06582  2.42729E-04 0.11524  5.53693E-05 0.25968 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.91635E-01 0.09372  1.24746E-02 0.00015  3.23429E-02 0.00053  1.06630E-01 0.00366  2.97757E-01 0.00159  1.23693E+00 0.00099  6.46151E+00 0.10693 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.89886E+03 0.03954 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.18971E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.16302E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18763E-03 0.00586 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.89334E+03 0.00594 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16106E-09 0.00113 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.18017E-05 0.00458  7.18158E-05 0.00459  9.26276E-07 0.41545 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.68440E-05 0.01648  7.68502E-05 0.01650  1.33627E-06 0.53197 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.21222E-04 0.01446  4.21978E-04 0.01445  1.95729E-04 0.40791 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70158E+01 0.01197 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.43298E+01 0.00027  4.02108E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 19:22:48 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02124E+00  1.00330E+00  1.00027E+00  9.96159E-01  9.92868E-01  1.00352E+00  9.97570E-01  9.95833E-01  9.91385E-01  1.00353E+00  9.95885E-01  1.00164E+00  1.00327E+00  9.88551E-01  9.97648E-01  1.00234E+00  9.93958E-01  1.01104E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 9.3E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.57095E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74290E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39682E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53471E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59577E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45652E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44721E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.13442E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.11688E-01 0.00057  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500730 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50035E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50035E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.41114E+02 ;
RUNNING_TIME              (idx, 1)        =  8.03190E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.71900E-01  8.81667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.90100E+01  3.13402E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.17667E-02  1.53333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.71982E+00  2.49370E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.78749E+01  4.33542E+02 ];
CPU_USAGE                 (idx, 1)        = 9.22714 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98487E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.16063E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 173 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.34443E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.64960E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.21594E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.82050E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.27202E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49322E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49161E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.68803E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.38125E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.79306E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.82094E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89475E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.59915E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49249E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34256E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33140E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53962E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.71742E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17708E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.97693E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.50217E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.32129E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.42529E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.61453E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45044E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.77365E-04 -1.27480E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.80559E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.55431E+18 0.00399  1.65696E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  9.21257E+19 0.00050  9.82129E-01 6.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.07406E+16 0.04642  1.14561E-04 0.04642 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26387E+19 0.00053  7.44482E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34242E+19 0.00134  1.07884E-01 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  3.77397E+15 0.08104  3.03203E-05 0.08098 ];
SM149_CAPT                (idx, [1:   4]) = [  5.78629E+16 0.02068  4.64947E-04 0.02064 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001409 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16852E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001409 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5171473 5.17695E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3898603 3.90256E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931333 9.32170E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001409 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.36556E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36442E+20 3.8E-06  2.36442E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37756E+19 1.2E-07  9.37756E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.24477E+20 0.00030  1.11198E+20 0.00021  1.32791E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.18253E+20 0.00017  2.04974E+20 0.00012  1.32791E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40363E+20 0.00038  2.40363E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.16984E+22 0.00027  3.00862E+22 0.00024  1.61226E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24072E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.40660E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07012E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51429E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51429E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.83770E-01 0.05214 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.00770E-01 0.01080 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.71810E-04 0.02089 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.64913E+03 0.02276 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06787E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.73025E-01 0.03920 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.10300E-01 0.03920 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52136E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99674E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.83773E-01 0.00042  6.13013E-02 0.00040  1.96569E-04 0.00951 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.83625E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.83743E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.83625E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08473E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03544E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03570E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32067E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32010E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49476E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49668E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28382E-03 0.00578  2.70340E-04 0.02022  7.97096E-04 0.01140  6.07938E-04 0.01277  1.28887E-03 0.00919  2.68352E-04 0.02005  5.12227E-05 0.04509 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45773E-01 0.01536  1.24758E-02 2.9E-05  3.23306E-02 9.6E-05  1.05969E-01 0.00059  2.97363E-01 0.00025  1.23299E+00 0.00252  4.55436E+00 0.04158 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20511E-03 0.00828  2.61672E-04 0.03051  7.68882E-04 0.01693  5.96164E-04 0.01960  1.26280E-03 0.01297  2.67912E-04 0.02985  4.76835E-05 0.06813 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42549E-01 0.02128  1.24762E-02 3.4E-05  3.23318E-02 0.00013  1.05977E-01 0.00079  2.97327E-01 0.00038  1.23636E+00 0.00028  6.42433E+00 0.02840 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.41274E-07 0.00127  8.40983E-07 0.00128  9.25708E-07 0.01691 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.27567E-07 0.00120  8.27280E-07 0.00121  9.10635E-07 0.01689 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21402E-03 0.00966  2.59118E-04 0.03108  7.79680E-04 0.01929  6.04772E-04 0.02107  1.25908E-03 0.01560  2.59040E-04 0.03177  5.23309E-05 0.07335 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.39647E-01 0.02554  1.24760E-02 4.7E-05  3.23313E-02 0.00018  1.05908E-01 0.00097  2.97307E-01 0.00044  1.23636E+00 0.00039  5.90040E+00 0.04348 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.49339E-07 0.01712  7.48848E-07 0.01712  8.23905E-07 0.05071 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.37471E-07 0.01711  7.36988E-07 0.01711  8.10939E-07 0.05072 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.90955E-03 0.04402  2.39463E-04 0.14541  6.92708E-04 0.07996  5.45757E-04 0.09160  1.13215E-03 0.06293  2.46441E-04 0.14485  5.30339E-05 0.28440 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.40397E-01 0.08291  1.24748E-02 0.00015  3.23054E-02 0.00036  1.06300E-01 0.00336  2.97361E-01 0.00155  1.23674E+00 0.00124  6.06329E+00 0.11435 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91464E-03 0.04282  2.36104E-04 0.14094  7.01645E-04 0.08047  5.58702E-04 0.08772  1.12452E-03 0.06143  2.41353E-04 0.14424  5.23160E-05 0.26840 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.37074E-01 0.07931  1.24750E-02 0.00015  3.23071E-02 0.00037  1.06274E-01 0.00332  2.97359E-01 0.00154  1.23678E+00 0.00124  6.06329E+00 0.11435 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.90055E+03 0.04065 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.36150E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.22518E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19392E-03 0.00583 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82032E+03 0.00584 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.96843E-09 0.00100 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.91555E-05 0.00609  6.91495E-05 0.00610  1.84807E-06 0.30569 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.68218E-05 0.02302  6.68749E-05 0.02320  2.18939E-06 0.46381 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.59571E-04 0.02020  2.59321E-04 0.02033  3.41910E-04 0.29978 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70897E+01 0.01087 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44721E+01 0.00026  4.03980E+01 0.00036 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 19:54:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02483E+00  1.00127E+00  1.00054E+00  9.91870E-01  9.97742E-01  9.98330E-01  1.00178E+00  9.97488E-01  9.95293E-01  9.96344E-01  9.92621E-01  1.00760E+00  9.97899E-01  1.00007E+00  1.00139E+00  1.00401E+00  9.84645E-01  1.00629E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.56539E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74346E-01 8.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39748E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53510E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59014E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46976E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46049E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.15730E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.10678E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500226 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50023E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50023E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.03697E+03 ;
RUNNING_TIME              (idx, 1)        =  1.12356E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.57733E-01  8.58333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.10936E+02  3.19261E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.55667E-02  1.38000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.35103E+00  2.63115E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.09774E+02  4.55612E+02 ];
CPU_USAGE                 (idx, 1)        = 9.22932 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98501E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.16201E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 173 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.45991E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66465E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.07737E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85894E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29797E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.56989E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50371E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.74032E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.54163E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.94871E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.01832E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79053E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.73977E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48747E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.35669E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34222E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55416E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.40192E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42514E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.04856E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52828E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.10789E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.58578E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.70800E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.13309E-03 -2.50182E+26  2.21047E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.96560E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.57157E+18 0.00405  1.67517E-02 0.00397 ];
U233_FISS                 (idx, [1:   4]) = [  9.19849E+19 0.00047  9.80567E-01 7.4E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.01920E+16 0.02446  4.28477E-04 0.02442 ];
TH232_CAPT                (idx, [1:   4]) = [  9.39281E+19 0.00055  7.41608E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34231E+19 0.00122  1.05985E-01 0.00119 ];
U235_CAPT                 (idx, [1:   4]) = [  1.44711E+16 0.04036  1.14318E-04 0.04044 ];
SM149_CAPT                (idx, [1:   4]) = [  9.61065E+16 0.01577  7.58983E-04 0.01580 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000915 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15220E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000915 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5212841 5.21857E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3861275 3.86524E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 926799 9.27710E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000915 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.19095E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36435E+20 3.6E-06  2.36435E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37749E+19 1.2E-07  9.37749E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26595E+20 0.00032  1.13116E+20 0.00023  1.34795E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20370E+20 0.00018  2.06891E+20 0.00012  1.34795E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42700E+20 0.00037  2.42700E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.20611E+22 0.00028  3.04300E+22 0.00024  1.63113E+21 0.00187 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.25163E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42887E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08373E+22 0.00037 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51428E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51428E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.41391E-01 0.04828 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.16470E-01 0.01238 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.25522E-04 0.02144 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.41756E+03 0.02283 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07232E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.06300E-01 0.03614 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.40772E-01 0.03614 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52131E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.74355E-01 0.00041  6.07109E-02 0.00040  1.96962E-04 0.00847 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.74584E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.74239E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.74584E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07428E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03920E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03823E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31571E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31677E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50507E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50710E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.32151E-03 0.00546  2.70493E-04 0.02002  8.24038E-04 0.01092  6.10532E-04 0.01283  1.30062E-03 0.00876  2.62892E-04 0.01999  5.29339E-05 0.04761 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43294E-01 0.01676  1.24439E-02 0.00251  3.23321E-02 8.7E-05  1.06025E-01 0.00060  2.97411E-01 0.00026  1.23564E+00 0.00026  4.31210E+00 0.04320 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25812E-03 0.00807  2.60483E-04 0.02892  8.19593E-04 0.01587  6.04850E-04 0.01934  1.26464E-03 0.01334  2.56138E-04 0.02866  5.24129E-05 0.06714 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.40496E-01 0.02215  1.24755E-02 3.9E-05  3.23289E-02 0.00012  1.06021E-01 0.00084  2.97467E-01 0.00037  1.23584E+00 0.00035  6.22053E+00 0.02954 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.49584E-07 0.00124  8.49258E-07 0.00124  9.51427E-07 0.01707 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.27738E-07 0.00116  8.27421E-07 0.00117  9.26976E-07 0.01706 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24442E-03 0.00867  2.64757E-04 0.03081  8.04135E-04 0.01719  5.99569E-04 0.02240  1.27680E-03 0.01409  2.50968E-04 0.03469  4.81953E-05 0.07360 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.35440E-01 0.02458  1.24749E-02 5.2E-05  3.23235E-02 0.00015  1.05864E-01 0.00098  2.97315E-01 0.00043  1.23598E+00 0.00052  6.43413E+00 0.04128 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.51998E-07 0.01708  7.51593E-07 0.01708  8.26958E-07 0.05599 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.32918E-07 0.01708  7.32524E-07 0.01708  8.05851E-07 0.05590 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98034E-03 0.04002  2.74007E-04 0.12433  8.48504E-04 0.07483  5.44504E-04 0.09392  1.01390E-03 0.06294  2.80366E-04 0.13033  1.90580E-05 0.38575 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.85978E-01 0.07146  1.24723E-02 0.00019  3.23369E-02 0.00054  1.06016E-01 0.00307  2.97768E-01 0.00164  1.23733E+00 0.00137  5.81117E+00 0.18149 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97826E-03 0.03911  2.72882E-04 0.12083  8.58617E-04 0.07408  5.49764E-04 0.09168  1.00485E-03 0.06168  2.74059E-04 0.12780  1.80822E-05 0.37261 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.81209E-01 0.06566  1.24723E-02 0.00019  3.23350E-02 0.00054  1.06026E-01 0.00309  2.97768E-01 0.00164  1.23726E+00 0.00137  5.81117E+00 0.18149 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.98767E+03 0.03642 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.44986E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.23254E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23453E-03 0.00585 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82892E+03 0.00592 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92204E-09 0.00088 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79037E-05 0.00792  6.79162E-05 0.00792  4.66084E-07 0.57671 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.44869E-05 0.02599  6.44490E-05 0.02602  5.50302E-07 0.72575 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.15598E-04 0.02064  2.16011E-04 0.02063  9.66980E-05 0.57743 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73365E+01 0.01179 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46049E+01 0.00026  4.05079E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 20:28:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01685E+00  9.99869E-01  9.97191E-01  1.00677E+00  9.98262E-01  1.00038E+00  9.95871E-01  1.00204E+00  9.98765E-01  1.00218E+00  9.99419E-01  9.95074E-01  9.98537E-01  9.95381E-01  1.00035E+00  1.00391E+00  9.91312E-01  9.97831E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.55779E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74422E-01 8.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39551E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53279E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58859E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48766E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47836E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19826E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.11593E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2501001 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50067E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50067E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.34186E+03 ;
RUNNING_TIME              (idx, 1)        =  1.45625E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.48950E-01  9.12167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.44085E+02  3.31493E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.99500E-02  1.43833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.13427E+01  2.99158E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.42687E+02  4.63004E+02 ];
CPU_USAGE                 (idx, 1)        = 9.21446 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98520E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.14494E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 173 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.48616E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65808E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.45972E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88547E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31496E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.56931E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49516E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.92017E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.66101E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.22530E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.13710E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.68924E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.84718E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93401E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.31595E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31092E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51121E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35710E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81237E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.04539E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.49630E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.48666E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.63660E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.84713E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.31031E-03 -5.10109E+26  2.21306E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.21975E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.59677E+18 0.00378  1.70282E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  9.16111E+19 0.00051  9.76930E-01 7.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.55715E+17 0.01247  1.66027E-03 0.01242 ];
TH232_CAPT                (idx, [1:   4]) = [  9.57390E+19 0.00049  7.37830E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33706E+19 0.00128  1.03043E-01 0.00121 ];
U235_CAPT                 (idx, [1:   4]) = [  5.51529E+16 0.02115  4.24921E-04 0.02113 ];
SM149_CAPT                (idx, [1:   4]) = [  1.34320E+17 0.01346  1.03516E-03 0.01346 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002661 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14528E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002661 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5266186 5.27094E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3805931 3.80923E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 930544 9.31282E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002661 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.80677E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36419E+20 3.8E-06  2.36419E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29764E+20 0.00028  1.16068E+20 0.00020  1.36964E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23537E+20 0.00016  2.09841E+20 0.00011  1.36964E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46178E+20 0.00033  2.46178E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.26154E+22 0.00023  3.09624E+22 0.00022  1.65298E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.29273E+19 0.00156 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46464E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10367E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51149E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51149E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.85292E-01 0.05023 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33531E-01 0.01297 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.07818E-04 0.02405 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.15514E+03 0.02411 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06874E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.80412E-01 0.03735 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.17096E-01 0.03735 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52119E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99680E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.60389E-01 0.00042  5.98268E-02 0.00041  1.96111E-04 0.00914 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.60363E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.60399E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.60363E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05898E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04330E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04434E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31029E-02 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30872E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52794E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52613E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.42568E-03 0.00535  2.82553E-04 0.01886  8.44416E-04 0.01082  6.25496E-04 0.01317  1.33254E-03 0.00888  2.85558E-04 0.01926  5.51171E-05 0.04403 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.52783E-01 0.01507  1.24764E-02 2.6E-05  3.23304E-02 9.5E-05  1.06211E-01 0.00060  2.97625E-01 0.00029  1.23225E+00 0.00252  4.85199E+00 0.03998 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27719E-03 0.00791  2.65058E-04 0.02709  7.95231E-04 0.01585  6.21238E-04 0.02016  1.27083E-03 0.01339  2.70757E-04 0.02976  5.40709E-05 0.06517 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.54047E-01 0.02264  1.24765E-02 3.2E-05  3.23316E-02 0.00014  1.06163E-01 0.00083  2.97525E-01 0.00041  1.23529E+00 0.00035  6.65936E+00 0.02691 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.66997E-07 0.00129  8.66648E-07 0.00129  9.72528E-07 0.01631 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32587E-07 0.00120  8.32251E-07 0.00120  9.34232E-07 0.01635 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26329E-03 0.00920  2.77489E-04 0.03317  7.98868E-04 0.01725  5.95558E-04 0.02041  1.26621E-03 0.01474  2.76737E-04 0.03207  4.84242E-05 0.07186 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.43768E-01 0.02307  1.24765E-02 4.0E-05  3.23311E-02 0.00015  1.06270E-01 0.00103  2.97539E-01 0.00047  1.23494E+00 0.00042  6.61037E+00 0.03993 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.65721E-07 0.01706  7.65637E-07 0.01706  7.71602E-07 0.05628 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.35618E-07 0.01705  7.35537E-07 0.01705  7.41584E-07 0.05633 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08123E-03 0.03972  2.79478E-04 0.11731  7.27480E-04 0.07603  5.12042E-04 0.09156  1.23324E-03 0.06161  2.80309E-04 0.11453  4.86768E-05 0.31857 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52115E-01 0.09327  1.24757E-02 0.00012  3.23410E-02 0.00058  1.06196E-01 0.00328  2.97457E-01 0.00152  1.23617E+00 0.00102  6.28702E+00 0.12643 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06455E-03 0.03914  2.80376E-04 0.11521  7.27243E-04 0.07653  5.23107E-04 0.08947  1.19752E-03 0.05975  2.86959E-04 0.11291  4.93459E-05 0.31664 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.56703E-01 0.08944  1.24756E-02 0.00012  3.23413E-02 0.00058  1.06185E-01 0.00330  2.97540E-01 0.00154  1.23615E+00 0.00103  6.28499E+00 0.12651 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.04264E+03 0.03585 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60760E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.26605E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32108E-03 0.00583 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.85856E+03 0.00581 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91902E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63285E-05 0.00895  6.63715E-05 0.00897  8.55141E-07 0.41452 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.91086E-05 0.02766  5.91321E-05 0.02766  7.49362E-07 0.56624 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.00780E-04 0.02353  2.00869E-04 0.02347  1.81163E-04 0.40974 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71053E+01 0.01112 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47836E+01 0.00026  4.06568E+01 0.00033 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 21:01:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01709E+00  1.00086E+00  9.97115E-01  9.99512E-01  1.00575E+00  9.96128E-01  1.00180E+00  9.98643E-01  1.00538E+00  9.97735E-01  9.95442E-01  1.00417E+00  9.91516E-01  1.00079E+00  1.00069E+00  9.97859E-01  9.86453E-01  1.00307E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.55508E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74449E-01 8.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39269E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52989E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58713E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50147E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49216E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23394E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.13053E-01 0.00057  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500640 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50060E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50060E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.65048E+03 ;
RUNNING_TIME              (idx, 1)        =  1.79255E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.38600E-01  8.96500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.77590E+02  3.35045E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.47000E-02  1.47500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.43114E+01  2.96865E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.76337E+02  4.75620E+02 ];
CPU_USAGE                 (idx, 1)        = 9.20746 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98499E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.13685E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 175 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.55925E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66826E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.92978E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92487E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34122E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.60254E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50226E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.08962E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.77347E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.63712E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.31638E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.24455E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94154E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34753E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32688E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31888E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52208E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02994E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.16927E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08860E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52972E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.28750E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.74771E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.96763E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85969E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.46965E-03 -7.66087E+26  2.21562E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.44459E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59498E+18 0.00399  1.70017E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  9.12800E+19 0.00052  9.73072E-01 8.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.31804E+17 0.00867  3.53694E-03 0.00864 ];
TH232_CAPT                (idx, [1:   4]) = [  9.73584E+19 0.00054  7.35101E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33100E+19 0.00136  1.00500E-01 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.17580E+17 0.01494  8.87908E-04 0.01494 ];
SM149_CAPT                (idx, [1:   4]) = [  1.50387E+17 0.01312  1.13568E-03 0.01315 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002387 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16474E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002387 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5309865 5.31492E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3761008 3.76444E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931514 9.32287E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002387 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.42727E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36403E+20 3.7E-06  2.36403E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37701E+19 1.2E-07  9.37701E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32459E+20 0.00029  1.18566E+20 0.00021  1.38924E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26229E+20 0.00017  2.12336E+20 0.00012  1.38924E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49191E+20 0.00036  2.49191E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.31051E+22 0.00024  3.14316E+22 0.00022  1.67342E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.32325E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.49461E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12063E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50737E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50737E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.53543E-01 0.05288 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.18572E-01 0.01347 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90141E-04 0.02331 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.34758E+03 0.02320 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06774E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.22873E-01 0.04131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.64788E-01 0.04131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52109E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.48926E-01 0.00041  5.91245E-02 0.00040  1.91000E-04 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.48766E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.48730E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.48766E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04632E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04525E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04496E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30783E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30791E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54400E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54816E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41593E-03 0.00597  2.72303E-04 0.01920  8.32058E-04 0.01160  6.37741E-04 0.01294  1.33410E-03 0.00931  2.84930E-04 0.01875  5.48037E-05 0.04214 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.52442E-01 0.01464  1.24757E-02 2.8E-05  3.23282E-02 0.00010  1.06071E-01 0.00054  2.97748E-01 0.00027  1.23372E+00 0.00253  4.74732E+00 0.03981 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21558E-03 0.00802  2.60846E-04 0.02855  7.97411E-04 0.01769  6.01553E-04 0.01823  1.24110E-03 0.01310  2.68189E-04 0.02800  4.64864E-05 0.06772 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42511E-01 0.02250  1.24756E-02 3.7E-05  3.23260E-02 0.00014  1.06055E-01 0.00076  2.97614E-01 0.00036  1.23704E+00 0.00037  6.50531E+00 0.02727 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82386E-07 0.00129  8.82011E-07 0.00129  9.95680E-07 0.01778 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.37259E-07 0.00121  8.36903E-07 0.00121  9.44854E-07 0.01783 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23058E-03 0.00892  2.48539E-04 0.03328  7.64865E-04 0.01797  6.13338E-04 0.02130  1.28527E-03 0.01506  2.68964E-04 0.03153  4.96110E-05 0.07536 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47103E-01 0.02444  1.24751E-02 5.9E-05  3.23274E-02 0.00017  1.06149E-01 0.00099  2.97765E-01 0.00047  1.23679E+00 0.00056  6.35818E+00 0.04174 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.84865E-07 0.01715  7.84805E-07 0.01715  7.57184E-07 0.05376 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.45007E-07 0.01715  7.44950E-07 0.01715  7.18793E-07 0.05373 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94947E-03 0.04277  1.99286E-04 0.13916  7.06787E-04 0.08371  5.87503E-04 0.08989  1.22774E-03 0.06106  1.91020E-04 0.14459  3.71346E-05 0.32937 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.79352E-01 0.13506  1.24764E-02 0.00014  3.23190E-02 0.00052  1.05418E-01 0.00199  2.98635E-01 0.00176  1.23659E+00 0.00127  7.10328E+00 0.11140 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94558E-03 0.04216  1.97830E-04 0.13836  7.16675E-04 0.08116  5.72316E-04 0.08582  1.23078E-03 0.06008  1.92723E-04 0.14623  3.52538E-05 0.33341 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.70100E-01 0.13014  1.24764E-02 0.00014  3.23213E-02 0.00052  1.05408E-01 0.00199  2.98616E-01 0.00174  1.23660E+00 0.00127  7.10328E+00 0.11140 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77819E+03 0.03973 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.78555E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33625E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23138E-03 0.00553 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.67891E+03 0.00559 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91576E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79104E-05 0.00870  6.79436E-05 0.00874  5.69421E-07 0.51131 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.56972E-05 0.02639  5.55504E-05 0.02612  1.10830E-06 0.73741 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.81129E-04 0.02265  1.81348E-04 0.02274  1.32479E-04 0.49902 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71093E+01 0.01136 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49216E+01 0.00027  4.07330E+01 0.00037 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 21:35:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01311E+00  1.00024E+00  1.00876E+00  9.99176E-01  9.95164E-01  1.00425E+00  9.90917E-01  1.00006E+00  9.95974E-01  9.94328E-01  1.00269E+00  1.00118E+00  9.96340E-01  1.00002E+00  9.94471E-01  1.00456E+00  9.98078E-01  1.00068E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 6.6E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.55399E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74460E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38961E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52681E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58909E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52131E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51200E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.28148E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.16251E-01 0.00057  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500251 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50055E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50055E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.96613E+03 ;
RUNNING_TIME              (idx, 1)        =  2.13254E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.27200E-01  8.86000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.11448E+02  3.38580E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.06800E-01  3.21000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.68897E+01  2.57830E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.10723E+02  4.79724E+02 ];
CPU_USAGE                 (idx, 1)        = 9.21968 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98406E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.15117E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 176 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.58722E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66470E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.99523E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95872E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36219E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.59620E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49618E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26083E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.90585E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.64333E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.49756E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.91983E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.05521E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.05984E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.29314E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29199E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.48579E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.84430E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.77719E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09357E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53164E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.81548E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.80100E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.01416E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43282E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.89930E-03 -1.30254E+27  2.22099E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.76836E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.63213E+18 0.00392  1.73899E-02 0.00383 ];
U233_FISS                 (idx, [1:   4]) = [  9.04386E+19 0.00053  9.63683E-01 9.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.16391E+17 0.00548  8.69947E-03 0.00547 ];
PU239_FISS                (idx, [1:   4]) = [  2.55945E+13 1.00000  2.68240E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  9.93836E+19 0.00051  7.28840E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32043E+19 0.00142  9.68342E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  2.89421E+17 0.00888  2.12263E-03 0.00888 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71425E+17 0.01165  1.25708E-03 0.01163 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002214 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15084E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002214 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5372960 5.37826E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3698248 3.70150E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931006 9.31750E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002214 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.31323E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36363E+20 3.8E-06  2.36363E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37629E+19 1.2E-07  9.37629E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36430E+20 0.00030  1.22119E+20 0.00021  1.43110E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30193E+20 0.00018  2.15882E+20 0.00012  1.43110E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53540E+20 0.00037  2.53540E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.38073E+22 0.00027  3.20896E+22 0.00025  1.71770E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36243E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53817E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14521E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49923E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49923E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.36819E-01 0.04713 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.44748E-01 0.01156 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.91991E-04 0.02304 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.13776E+03 0.02313 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06827E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.86934E-01 0.03534 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.22869E-01 0.03534 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52086E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.32954E-01 0.00044  5.81254E-02 0.00042  1.91332E-04 0.00909 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.32320E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.32304E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.32320E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02812E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04456E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04347E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30867E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30988E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.58592E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.58903E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.58403E-03 0.00590  2.91023E-04 0.01976  8.71999E-04 0.01125  6.54703E-04 0.01267  1.40396E-03 0.00901  3.06134E-04 0.01914  5.62127E-05 0.04206 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.52986E-01 0.01436  1.24448E-02 0.00251  3.23160E-02 0.00010  1.06146E-01 0.00055  2.98063E-01 0.00028  1.23853E+00 0.00033  4.83516E+00 0.03927 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.31414E-03 0.00880  2.74515E-04 0.03031  8.12030E-04 0.01663  6.17317E-04 0.01933  1.28924E-03 0.01327  2.74195E-04 0.02768  4.68389E-05 0.06243 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.37299E-01 0.01956  1.24760E-02 3.9E-05  3.23153E-02 0.00014  1.06140E-01 0.00081  2.98068E-01 0.00041  1.23879E+00 0.00044  6.56248E+00 0.02723 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.01637E-07 0.00123  9.01323E-07 0.00123  9.97753E-07 0.01937 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.41127E-07 0.00116  8.40835E-07 0.00117  9.30665E-07 0.01933 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27828E-03 0.00923  2.71660E-04 0.03230  7.93002E-04 0.01852  5.93913E-04 0.02027  1.28951E-03 0.01572  2.74905E-04 0.03009  5.52890E-05 0.07051 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.63240E-01 0.02608  1.24754E-02 5.3E-05  3.23076E-02 0.00016  1.06170E-01 0.00111  2.98093E-01 0.00048  1.23945E+00 0.00073  6.55580E+00 0.03788 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.94790E-07 0.01715  7.94633E-07 0.01716  8.15152E-07 0.04767 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.41495E-07 0.01714  7.41348E-07 0.01714  7.60382E-07 0.04765 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11201E-03 0.04081  2.80970E-04 0.12756  7.48773E-04 0.08268  6.21906E-04 0.08267  1.18163E-03 0.06428  2.29431E-04 0.13735  4.92978E-05 0.33642 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.33182E-01 0.08425  1.24775E-02 9.3E-05  3.23556E-02 0.00074  1.05636E-01 0.00249  2.98935E-01 0.00176  1.23942E+00 0.00222  5.85587E+00 0.13524 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10499E-03 0.04030  2.84515E-04 0.12824  7.40843E-04 0.08190  6.19808E-04 0.08314  1.18070E-03 0.06313  2.33921E-04 0.13698  4.52008E-05 0.33210 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.29388E-01 0.08185  1.24775E-02 9.4E-05  3.23547E-02 0.00074  1.05637E-01 0.00247  2.98953E-01 0.00177  1.23949E+00 0.00224  5.85587E+00 0.13524 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.94833E+03 0.03748 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.93367E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33405E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25673E-03 0.00708 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.64559E+03 0.00705 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95398E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68477E-05 0.00782  6.68647E-05 0.00778  9.50866E-07 0.43061 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.99916E-05 0.02485  5.99934E-05 0.02484  7.48664E-07 0.65483 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.83987E-04 0.02224  1.84052E-04 0.02239  1.83840E-04 0.41164 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70022E+01 0.01143 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51200E+01 0.00025  4.07989E+01 0.00034 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 22:10:14 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01830E+00  1.00401E+00  1.00144E+00  1.00787E+00  9.99984E-01  9.93496E-01  9.93588E-01  9.93953E-01  9.94300E-01  1.00759E+00  9.94587E-01  1.00430E+00  9.98318E-01  1.00187E+00  9.95933E-01  1.00003E+00  9.97690E-01  9.92745E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.55622E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74438E-01 8.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38442E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52190E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59006E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52931E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51997E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.31402E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.18419E-01 0.00057  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500151 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50023E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50023E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.28587E+03 ;
RUNNING_TIME              (idx, 1)        =  2.47758E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.19133E-01  9.19333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.45822E+02  3.43742E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.21683E-01  1.48833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.95911E+01  2.70135E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.45106E+02  4.82860E+02 ];
CPU_USAGE                 (idx, 1)        = 9.22619 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98537E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.15817E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 178 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.67053E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67672E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.98770E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99841E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38448E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.63911E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50532E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.74350E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.20650E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.07457E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.79559E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.19876E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.32376E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32644E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.28392E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.28142E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.47329E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.55274E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27677E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13407E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.63089E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.83474E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.90347E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03464E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.21209E-02 -2.67625E+27  2.23473E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01939E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.63419E+18 0.00417  1.74360E-02 0.00412 ];
U233_FISS                 (idx, [1:   4]) = [  8.80974E+19 0.00055  9.39977E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.28985E+18 0.00343  2.44323E-02 0.00340 ];
PU239_FISS                (idx, [1:   4]) = [  9.80560E+14 0.15879  1.04535E-05 0.15890 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01021E+20 0.00053  7.16211E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28838E+19 0.00145  9.13435E-02 0.00140 ];
U235_CAPT                 (idx, [1:   4]) = [  8.23972E+17 0.00553  5.84221E-03 0.00554 ];
PU239_CAPT                (idx, [1:   4]) = [  3.87221E+14 0.25365  2.74379E-06 0.25365 ];
PU240_CAPT                (idx, [1:   4]) = [  2.63473E+13 1.00000  1.86637E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71225E+17 0.01194  1.21421E-03 0.01196 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000930 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16056E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000930 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5447198 5.45310E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3619582 3.62342E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 934150 9.35094E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000930 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.47035E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36263E+20 4.0E-06  2.36263E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37411E+19 1.1E-07  9.37411E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.41041E+20 0.00031  1.26438E+20 0.00022  1.46037E+19 0.00201 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34783E+20 0.00019  2.20179E+20 0.00012  1.46037E+19 0.00201 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58661E+20 0.00036  2.58661E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45698E+22 0.00025  3.28211E+22 0.00022  1.74875E+21 0.00192 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41881E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58971E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17043E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47879E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47879E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.82253E-01 0.05401 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.03756E-01 0.01525 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.81731E-04 0.02458 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.06349E+03 0.02330 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06494E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.89158E-01 0.04217 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.34107E-01 0.04217 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52038E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99747E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.13159E-01 0.00045  5.68895E-02 0.00043  1.89103E-04 0.00890 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.13374E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.13456E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.13374E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00759E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01959E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02065E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34184E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34011E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.68751E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.68480E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.67660E-03 0.00569  2.86276E-04 0.01922  8.81489E-04 0.01159  6.64680E-04 0.01265  1.46495E-03 0.00852  3.17253E-04 0.01856  6.19453E-05 0.04425 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.68365E-01 0.01480  1.24763E-02 3.2E-05  3.23008E-02 0.00011  1.06379E-01 0.00059  2.98859E-01 0.00028  1.24490E+00 0.00055  4.98737E+00 0.03758 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35419E-03 0.00811  2.55410E-04 0.02970  7.95282E-04 0.01646  6.14857E-04 0.01890  1.34893E-03 0.01263  2.85992E-04 0.02581  5.37223E-05 0.06316 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.64622E-01 0.02097  1.24766E-02 3.7E-05  3.23016E-02 0.00015  1.06410E-01 0.00085  2.98886E-01 0.00041  1.24496E+00 0.00076  6.92322E+00 0.02383 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.15782E-07 0.00133  9.15387E-07 0.00134  1.03547E-06 0.01593 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.36187E-07 0.00125  8.35826E-07 0.00126  9.45615E-07 0.01595 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30709E-03 0.00897  2.54813E-04 0.03279  7.88830E-04 0.01899  5.95119E-04 0.02170  1.31732E-03 0.01473  2.93275E-04 0.03013  5.77362E-05 0.06884 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.78351E-01 0.02470  1.24767E-02 4.5E-05  3.22943E-02 0.00018  1.06362E-01 0.00111  2.98858E-01 0.00047  1.24554E+00 0.00100  7.06026E+00 0.03372 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.12689E-07 0.01721  8.12131E-07 0.01720  8.97857E-07 0.06437 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.42456E-07 0.01721  7.41944E-07 0.01720  8.20538E-07 0.06450 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.99178E-03 0.04087  2.40330E-04 0.13837  7.33023E-04 0.07919  5.57143E-04 0.09687  1.18233E-03 0.06182  2.20878E-04 0.13213  5.80784E-05 0.25433 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.72889E-01 0.10098  1.24777E-02 0.00011  3.23274E-02 0.00068  1.06214E-01 0.00311  2.99325E-01 0.00176  1.24509E+00 0.00274  5.40519E+00 0.11023 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.01384E-03 0.04030  2.48019E-04 0.13773  7.31720E-04 0.07839  5.49436E-04 0.09395  1.19846E-03 0.05983  2.26515E-04 0.12925  5.96796E-05 0.23532 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.81895E-01 0.10493  1.24777E-02 0.00011  3.23284E-02 0.00067  1.06220E-01 0.00311  2.99349E-01 0.00175  1.24506E+00 0.00274  5.40519E+00 0.11023 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.71472E+03 0.03779 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.09296E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.30269E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28736E-03 0.00540 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.61576E+03 0.00543 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.96360E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72168E-05 0.00964  6.72513E-05 0.00963  5.22829E-07 0.50311 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.53785E-05 0.02784  5.53526E-05 0.02788  7.38690E-07 0.65558 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.74924E-04 0.02424  1.75168E-04 0.02424  1.22259E-04 0.49915 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68233E+01 0.01151 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51997E+01 0.00028  4.06430E+01 0.00038 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 22:45:13 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02377E+00  1.00296E+00  9.96032E-01  9.97698E-01  1.00102E+00  9.97913E-01  1.00460E+00  9.96267E-01  1.00120E+00  9.94947E-01  9.96940E-01  9.95391E-01  1.00232E+00  9.96019E-01  9.93497E-01  1.00800E+00  9.89929E-01  1.00148E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.56457E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74354E-01 8.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38252E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52041E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58617E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51407E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50475E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.29139E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.19814E-01 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500004 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50034E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50034E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.60908E+03 ;
RUNNING_TIME              (idx, 1)        =  2.82745E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.12900E-01  9.37667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.80671E+02  3.48489E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.35667E-01  1.39833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.24344E+01  2.84322E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.79951E+02  4.86842E+02 ];
CPU_USAGE                 (idx, 1)        = 9.22766 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98498E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.15963E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 180 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.86569E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.72711E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.54848E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02262E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39563E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.80911E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55373E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.12023E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.76783E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.04137E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.22360E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04353E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.83851E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.37748E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.36888E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34042E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55743E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.83328E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30127E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.25656E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.90405E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.45373E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09800E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04182E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57630E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.29231E-02 -5.06134E+27  2.25858E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03213E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.60708E+18 0.00409  1.71604E-02 0.00407 ];
U233_FISS                 (idx, [1:   4]) = [  8.46350E+19 0.00053  9.03720E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.69216E+18 0.00237  5.01005E-02 0.00228 ];
PU239_FISS                (idx, [1:   4]) = [  2.01542E+16 0.03695  2.15301E-04 0.03700 ];
PU240_FISS                (idx, [1:   4]) = [  4.67443E+14 0.23065  4.98391E-06 0.23065 ];
PU241_FISS                (idx, [1:   4]) = [  5.75389E+14 0.22628  6.12210E-06 0.22618 ];
TH232_CAPT                (idx, [1:   4]) = [  9.91274E+19 0.00055  6.94304E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23717E+19 0.00142  8.66539E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.65930E+18 0.00401  1.16221E-02 0.00398 ];
U238_CAPT                 (idx, [1:   4]) = [  1.04762E+14 0.49812  7.34176E-07 0.49818 ];
PU239_CAPT                (idx, [1:   4]) = [  9.02946E+15 0.05454  6.32530E-05 0.05454 ];
PU240_CAPT                (idx, [1:   4]) = [  1.47906E+15 0.13230  1.03653E-05 0.13230 ];
PU241_CAPT                (idx, [1:   4]) = [  7.87082E+13 0.57592  5.48698E-07 0.57592 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77467E+17 0.01259  1.24296E-03 0.01258 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001350 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10672E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001350 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5476146 5.48168E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3592388 3.59574E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 932816 9.33651E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001350 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.31551E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36133E+20 3.9E-06  2.36133E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37027E+19 1.0E-07  9.37027E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42864E+20 0.00030  1.28205E+20 0.00022  1.46585E+19 0.00201 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36567E+20 0.00018  2.21908E+20 0.00012  1.46585E+19 0.00201 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60456E+20 0.00035  2.60456E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47465E+22 0.00026  3.29936E+22 0.00023  1.75296E+21 0.00194 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43187E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60885E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17453E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42132E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42132E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88125E-01 0.05172 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.51429E-01 0.01088 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.88425E-04 0.02353 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.91327E+03 0.02319 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06638E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.17123E-01 0.03941 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.59563E-01 0.03941 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52002E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99829E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05997E-01 0.00046  5.64403E-02 0.00045  1.93184E-04 0.00892 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06156E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06658E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06156E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99419E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.96461E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.96516E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.41765E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.41657E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81915E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81939E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.84743E-03 0.00568  2.89892E-04 0.01970  9.03628E-04 0.01118  6.97145E-04 0.01262  1.54716E-03 0.00851  3.32019E-04 0.01806  7.75848E-05 0.03812 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.96908E-01 0.01462  1.24762E-02 4.1E-05  3.22707E-02 0.00012  1.06522E-01 0.00057  2.99949E-01 0.00029  1.25365E+00 0.00068  5.77607E+00 0.02984 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44885E-03 0.00797  2.57272E-04 0.02935  7.97540E-04 0.01633  6.26700E-04 0.01804  1.40328E-03 0.01214  2.91924E-04 0.02602  7.21328E-05 0.05517 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00890E-01 0.02226  1.24759E-02 3.9E-05  3.22683E-02 0.00018  1.06488E-01 0.00078  2.99902E-01 0.00039  1.25451E+00 0.00094  6.89489E+00 0.02137 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.01820E-07 0.00119  9.01368E-07 0.00119  1.02936E-06 0.02016 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16999E-07 0.00116  8.16589E-07 0.00115  9.32671E-07 0.02031 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40122E-03 0.00905  2.71724E-04 0.03319  7.92911E-04 0.01934  6.11587E-04 0.02146  1.36180E-03 0.01415  2.95833E-04 0.03283  6.73608E-05 0.06652 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.92454E-01 0.02473  1.24759E-02 9.2E-05  3.22660E-02 0.00022  1.06585E-01 0.00097  2.99714E-01 0.00051  1.25718E+00 0.00138  6.85824E+00 0.03217 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.99652E-07 0.01718  7.99419E-07 0.01718  8.54458E-07 0.05632 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.24555E-07 0.01717  7.24346E-07 0.01718  7.73753E-07 0.05608 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91432E-03 0.04249  2.42386E-04 0.14374  6.77818E-04 0.08363  4.57445E-04 0.09949  1.15018E-03 0.06388  3.37481E-04 0.12806  4.90088E-05 0.29574 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.12446E-01 0.09420  1.24780E-02 0.00011  3.22691E-02 0.00060  1.07227E-01 0.00391  2.99297E-01 0.00179  1.26252E+00 0.00383  7.74445E+00 0.08356 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92667E-03 0.04161  2.47578E-04 0.14126  6.89026E-04 0.08228  4.54033E-04 0.09631  1.13857E-03 0.06187  3.42553E-04 0.12876  5.49104E-05 0.29135 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.23001E-01 0.09493  1.24780E-02 0.00011  3.22693E-02 0.00061  1.07232E-01 0.00389  2.99240E-01 0.00178  1.26219E+00 0.00378  7.75212E+00 0.08358 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.68674E+03 0.03980 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.96655E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12298E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35534E-03 0.00595 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.74287E+03 0.00599 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94430E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76820E-05 0.00807  6.77089E-05 0.00809  8.60183E-07 0.40817 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.67470E-05 0.02797  5.66838E-05 0.02806  8.75742E-07 0.56705 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.82154E-04 0.02287  1.82255E-04 0.02282  1.49573E-04 0.40633 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61605E+01 0.01121 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50475E+01 0.00027  4.01455E+01 0.00037 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 23:20:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01097E+00  9.99657E-01  9.98102E-01  9.95914E-01  9.95626E-01  9.93784E-01  9.98017E-01  1.00293E+00  9.98546E-01  1.00616E+00  9.99585E-01  1.00638E+00  1.00257E+00  9.97273E-01  1.00239E+00  1.00181E+00  1.00075E+00  9.89531E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.57013E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74299E-01 8.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38194E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52009E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58693E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50094E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49160E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26838E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.20386E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500306 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50042E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50042E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.93540E+03 ;
RUNNING_TIME              (idx, 1)        =  3.17700E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.08317E-01  9.54167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.15485E+02  3.48136E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.50000E-01  1.43333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.48919E+01  2.45740E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.15292E+02  4.90086E+02 ];
CPU_USAGE                 (idx, 1)        = 9.23952 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98501E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.17277E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 183 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.90957E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73865E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.61498E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02090E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39388E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85447E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56523E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.06973E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99039E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.89540E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.42559E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13397E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.03988E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81441E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38427E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35070E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57227E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88707E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73174E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28129E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97373E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.90537E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.13448E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04317E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21958E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.80392E-02 -6.19095E+27  2.26987E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03249E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.59830E+18 0.00442  1.70694E-02 0.00433 ];
U233_FISS                 (idx, [1:   4]) = [  8.35424E+19 0.00057  8.92310E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  5.39042E+18 0.00207  5.75761E-02 0.00203 ];
PU239_FISS                (idx, [1:   4]) = [  4.64301E+16 0.02437  4.95829E-04 0.02437 ];
PU240_FISS                (idx, [1:   4]) = [  1.79945E+15 0.12244  1.92237E-05 0.12236 ];
PU241_FISS                (idx, [1:   4]) = [  1.82811E+15 0.11778  1.95483E-05 0.11784 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81736E+19 0.00053  6.86226E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22313E+19 0.00151  8.54973E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  1.91655E+18 0.00359  1.33966E-02 0.00357 ];
U238_CAPT                 (idx, [1:   4]) = [  2.08384E+14 0.35046  1.45771E-06 0.35047 ];
PU239_CAPT                (idx, [1:   4]) = [  2.23669E+16 0.03434  1.56319E-04 0.03435 ];
PU240_CAPT                (idx, [1:   4]) = [  4.27349E+15 0.07724  2.99041E-05 0.07730 ];
PU241_CAPT                (idx, [1:   4]) = [  4.97026E+14 0.22422  3.45982E-06 0.22421 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83190E+17 0.01229  1.28059E-03 0.01230 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001693 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12000E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001693 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5480261 5.48578E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3586771 3.59005E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 934661 9.35375E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001693 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -8.00937E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36122E+20 4.3E-06  2.36122E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36885E+19 1.1E-07  9.36885E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42948E+20 0.00028  1.28341E+20 0.00020  1.46067E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36636E+20 0.00017  2.22030E+20 0.00011  1.46067E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60792E+20 0.00036  2.60792E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47260E+22 0.00025  3.29795E+22 0.00023  1.74646E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43950E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61031E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17262E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39043E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39043E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.31426E-01 0.05233 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.46260E-01 0.01149 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.87465E-04 0.02565 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.38137E+03 0.02152 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06465E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.01435E-01 0.04067 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.45287E-01 0.04067 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52029E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99860E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04759E-01 0.00045  5.63552E-02 0.00044  1.94119E-04 0.00842 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05593E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05451E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05593E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99062E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94016E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94065E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.45286E-02 0.00154 ];
IMP_EALF                  (idx, [1:   2]) = [  1.45174E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86860E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86828E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.87518E-03 0.00531  2.93321E-04 0.01863  9.08318E-04 0.01081  6.96285E-04 0.01223  1.55766E-03 0.00856  3.45071E-04 0.01780  7.45240E-05 0.03828 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.89091E-01 0.01385  1.24457E-02 0.00251  3.22589E-02 0.00012  1.06638E-01 0.00056  3.00449E-01 0.00028  1.25687E+00 0.00078  5.55465E+00 0.03186 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45262E-03 0.00793  2.48778E-04 0.03049  8.14533E-04 0.01625  6.26552E-04 0.01779  1.38632E-03 0.01256  3.10681E-04 0.02848  6.57496E-05 0.06019 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.86620E-01 0.02130  1.24769E-02 0.00010  3.22565E-02 0.00017  1.06675E-01 0.00080  3.00398E-01 0.00042  1.25696E+00 0.00104  6.75664E+00 0.02299 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.92088E-07 0.00141  8.91687E-07 0.00141  1.01034E-06 0.02063 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.07041E-07 0.00129  8.06679E-07 0.00129  9.13962E-07 0.02061 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43248E-03 0.00866  2.60217E-04 0.03327  8.13002E-04 0.01839  6.03490E-04 0.02039  1.39274E-03 0.01391  3.00392E-04 0.03010  6.26386E-05 0.06590 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.83017E-01 0.02549  1.24761E-02 8.4E-05  3.22486E-02 0.00022  1.06685E-01 0.00112  3.00534E-01 0.00050  1.25753E+00 0.00141  6.66480E+00 0.03372 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.84663E-07 0.01709  7.84533E-07 0.01709  8.23888E-07 0.05477 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.10106E-07 0.01709  7.09989E-07 0.01709  7.45852E-07 0.05484 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94591E-03 0.04261  2.35089E-04 0.15534  7.29929E-04 0.08353  4.95072E-04 0.09883  1.15411E-03 0.06639  2.56954E-04 0.14090  7.47541E-05 0.24414 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.19420E-01 0.10136  1.24780E-02 0.00018  3.22683E-02 0.00061  1.06476E-01 0.00331  3.00299E-01 0.00192  1.26727E+00 0.00425  7.33220E+00 0.08532 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94230E-03 0.04203  2.46285E-04 0.15586  7.23399E-04 0.08217  4.88853E-04 0.09878  1.15160E-03 0.06538  2.59203E-04 0.13168  7.29658E-05 0.24385 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.26225E-01 0.10168  1.24780E-02 0.00019  3.22668E-02 0.00061  1.06412E-01 0.00322  3.00249E-01 0.00192  1.26756E+00 0.00426  7.32928E+00 0.08539 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77145E+03 0.03913 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.84934E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.00576E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.41601E-03 0.00583 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.86152E+03 0.00593 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92887E-09 0.00085 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.59220E-05 0.01022  6.59448E-05 0.01025  1.40142E-06 0.33824 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.34947E-05 0.02720  5.35678E-05 0.02716  7.14300E-07 0.40143 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79222E-04 0.02501  1.78912E-04 0.02490  2.53958E-04 0.33609 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63594E+01 0.01153 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49160E+01 0.00027  3.99105E+01 0.00040 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Apr 17 23:54:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01455E+00  9.99010E-01  1.00298E+00  1.00079E+00  9.99794E-01  9.98683E-01  1.00130E+00  1.00261E+00  1.00215E+00  9.95763E-01  9.98389E-01  9.96455E-01  9.90882E-01  9.97886E-01  1.00074E+00  1.00372E+00  9.96514E-01  9.97782E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.57234E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74277E-01 8.1E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38181E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52008E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58846E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49377E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48443E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.25520E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.20108E-01 0.00058  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500522 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50069E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50069E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.26428E+03 ;
RUNNING_TIME              (idx, 1)        =  3.52487E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.00305E+00  9.47333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.50127E+02  3.46421E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.64833E-01  1.48333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.68746E+01  1.98263E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.50553E+02  4.90246E+02 ];
CPU_USAGE                 (idx, 1)        = 9.26072 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98632E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.19631E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 184 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.91449E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74036E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.45803E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00837E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38516E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87185E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56774E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.15317E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17436E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.89831E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.60575E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21133E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20520E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.19997E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37847E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34517E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56530E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.91434E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11446E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27746E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.00033E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.25809E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.12168E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04291E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86286E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.31843E-02 -7.32697E+27  2.28123E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02857E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.59091E+18 0.00407  1.69719E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  8.27788E+19 0.00055  8.83116E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  5.91730E+18 0.00220  6.31283E-02 0.00215 ];
PU239_FISS                (idx, [1:   4]) = [  8.90692E+16 0.01785  9.50252E-04 0.01783 ];
PU240_FISS                (idx, [1:   4]) = [  4.35746E+15 0.07477  4.64026E-05 0.07472 ];
PU241_FISS                (idx, [1:   4]) = [  5.03171E+15 0.07079  5.36548E-05 0.07075 ];
TH232_CAPT                (idx, [1:   4]) = [  9.70734E+19 0.00054  6.79279E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20771E+19 0.00154  8.45131E-02 0.00153 ];
U235_CAPT                 (idx, [1:   4]) = [  2.08260E+18 0.00351  1.45736E-02 0.00351 ];
U238_CAPT                 (idx, [1:   4]) = [  3.93130E+14 0.25365  2.74703E-06 0.25365 ];
PU239_CAPT                (idx, [1:   4]) = [  4.08412E+16 0.02626  2.85728E-04 0.02623 ];
PU240_CAPT                (idx, [1:   4]) = [  1.03070E+16 0.05107  7.21067E-05 0.05105 ];
PU241_CAPT                (idx, [1:   4]) = [  1.12053E+15 0.14798  7.83680E-06 0.14798 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78517E+17 0.01296  1.24952E-03 0.01300 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002777 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12566E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002777 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5476233 5.48110E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3592284 3.59515E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 934260 9.35011E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002777 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.62050E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36135E+20 4.6E-06  2.36135E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36759E+19 1.1E-07  9.36759E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42886E+20 0.00029  1.28284E+20 0.00020  1.46025E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36562E+20 0.00017  2.21960E+20 0.00012  1.46025E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60729E+20 0.00037  2.60729E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46860E+22 0.00027  3.29392E+22 0.00024  1.74687E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43798E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60942E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17046E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.35982E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.35982E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.24628E-01 0.04902 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27367E-01 0.01235 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.91567E-04 0.02390 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.10173E+03 0.02129 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06502E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.57177E-01 0.03614 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.95814E-01 0.03614 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52076E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99886E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.06101E-01 0.00044  5.64493E-02 0.00042  1.92010E-04 0.00901 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05953E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05722E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05953E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99428E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92189E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92097E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47956E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.48061E-02 0.00101 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.91230E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90863E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.87950E-03 0.00554  2.79889E-04 0.02016  9.15553E-04 0.01120  7.06025E-04 0.01245  1.54435E-03 0.00850  3.56801E-04 0.01739  7.68846E-05 0.03705 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.95612E-01 0.01418  1.24450E-02 0.00251  3.22495E-02 0.00012  1.06545E-01 0.00053  3.00635E-01 0.00033  1.25762E+00 0.00079  5.48323E+00 0.03043 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42588E-03 0.00749  2.40750E-04 0.03041  8.18701E-04 0.01650  6.22780E-04 0.01882  1.34437E-03 0.01270  3.29051E-04 0.02605  7.02289E-05 0.05613 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00505E-01 0.02028  1.24766E-02 3.6E-05  3.22486E-02 0.00019  1.06544E-01 0.00075  3.00651E-01 0.00047  1.25707E+00 0.00101  6.63810E+00 0.02200 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82048E-07 0.00129  8.81688E-07 0.00130  9.81782E-07 0.01475 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.99169E-07 0.00123  7.98842E-07 0.00123  8.89576E-07 0.01476 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39199E-03 0.00896  2.44236E-04 0.03428  7.90095E-04 0.01892  6.26824E-04 0.02084  1.35690E-03 0.01382  3.06803E-04 0.03059  6.71297E-05 0.06161 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.99092E-01 0.02566  1.24763E-02 5.1E-05  3.22358E-02 0.00023  1.06498E-01 0.00095  3.00767E-01 0.00057  1.25822E+00 0.00150  6.79776E+00 0.03125 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.71650E-07 0.01717  7.71252E-07 0.01717  8.39348E-07 0.05705 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.99321E-07 0.01715  6.98961E-07 0.01715  7.60771E-07 0.05703 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93275E-03 0.04243  2.41013E-04 0.13116  7.20360E-04 0.07407  5.01084E-04 0.09741  1.14472E-03 0.06506  2.64054E-04 0.13603  6.15215E-05 0.26681 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.09339E-01 0.11504  1.24766E-02 0.00014  3.22477E-02 0.00074  1.06767E-01 0.00337  3.00858E-01 0.00196  1.26339E+00 0.00456  6.56648E+00 0.09113 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91563E-03 0.04226  2.38435E-04 0.12916  7.05724E-04 0.07310  5.04374E-04 0.09592  1.14375E-03 0.06427  2.58551E-04 0.13469  6.47958E-05 0.26287 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.10980E-01 0.11387  1.24767E-02 0.00014  3.22483E-02 0.00074  1.06758E-01 0.00334  3.00889E-01 0.00195  1.26357E+00 0.00455  6.56648E+00 0.09113 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81961E+03 0.03898 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.74997E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.92777E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.41200E-03 0.00616 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90083E+03 0.00627 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92484E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.60225E-05 0.00864  6.60416E-05 0.00867  1.08443E-06 0.37770 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.94091E-05 0.02539  5.94297E-05 0.02547  1.03019E-06 0.44738 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84952E-04 0.02273  1.84972E-04 0.02274  1.78470E-04 0.38321 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61749E+01 0.01081 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48443E+01 0.00026  3.97517E+01 0.00038 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Apr 18 00:29:41 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01927E+00  1.00505E+00  9.99307E-01  1.00093E+00  9.92630E-01  1.00005E+00  9.92257E-01  1.00066E+00  9.99274E-01  9.94642E-01  9.97040E-01  1.00097E+00  9.95367E-01  1.00181E+00  1.00041E+00  9.95936E-01  9.98497E-01  1.00591E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.58688E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74131E-01 8.1E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37674E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51577E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59072E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47994E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47057E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24528E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.25423E-01 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500517 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50052E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50052E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.59638E+03 ;
RUNNING_TIME              (idx, 1)        =  3.87198E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.09678E+00  9.37333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.84694E+02  3.45676E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.80083E-01  1.52500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.84134E+01  1.53873E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.85705E+02  4.89958E+02 ];
CPU_USAGE                 (idx, 1)        = 9.28821 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98621E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.22691E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 187 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.93472E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74658E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.84660E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97107E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36267E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92942E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57627E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.71255E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75561E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.22086E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04173E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44252E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70418E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34183E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37408E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33901E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55850E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.96035E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27007E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27256E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05812E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.42599E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09404E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04590E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43599E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.36073E-02 -1.18363E+28  2.32633E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02763E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.55383E+18 0.00416  1.65962E-02 0.00411 ];
U233_FISS                 (idx, [1:   4]) = [  8.07008E+19 0.00057  8.61965E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.86445E+18 0.00197  7.33199E-02 0.00191 ];
U238_FISS                 (idx, [1:   4]) = [  2.67243E+13 1.00000  2.84770E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  3.01085E+17 0.00899  3.21607E-03 0.00899 ];
PU240_FISS                (idx, [1:   4]) = [  2.23776E+16 0.03348  2.38969E-04 0.03345 ];
PU241_FISS                (idx, [1:   4]) = [  3.30402E+16 0.02832  3.52991E-04 0.02832 ];
TH232_CAPT                (idx, [1:   4]) = [  9.50142E+19 0.00055  6.61541E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17810E+19 0.00147  8.20264E-02 0.00140 ];
U235_CAPT                 (idx, [1:   4]) = [  2.42615E+18 0.00320  1.68917E-02 0.00315 ];
U238_CAPT                 (idx, [1:   4]) = [  1.14853E+15 0.15269  7.99817E-06 0.15266 ];
PU239_CAPT                (idx, [1:   4]) = [  1.43435E+17 0.01326  9.98761E-04 0.01327 ];
PU240_CAPT                (idx, [1:   4]) = [  5.89336E+16 0.01977  4.10318E-04 0.01976 ];
PU241_CAPT                (idx, [1:   4]) = [  7.57736E+15 0.06114  5.27385E-05 0.06114 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83827E+17 0.01131  1.27999E-03 0.01132 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002089 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10519E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002089 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487763 5.49293E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3577602 3.58066E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 936724 9.37464E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002089 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.24797E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36289E+20 4.5E-06  2.36289E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36364E+19 1.0E-07  9.36364E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43596E+20 0.00029  1.28865E+20 0.00019  1.47312E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37233E+20 0.00017  2.22501E+20 0.00011  1.47312E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61475E+20 0.00034  2.61475E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47876E+22 0.00023  3.30299E+22 0.00021  1.75768E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.45134E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61746E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17017E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.24518E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.24518E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88910E-01 0.04942 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.25436E-01 0.01116 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.12990E-04 0.02135 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.78642E+03 0.02350 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06256E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.60475E-01 0.03574 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.98595E-01 0.03574 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52348E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99971E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03462E-01 0.00048  5.62754E-02 0.00048  1.96719E-04 0.00941 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.03746E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03721E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.03746E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97241E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.87846E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.87748E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.54518E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.54638E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.00329E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.00744E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96405E-03 0.00514  2.89229E-04 0.01980  9.23653E-04 0.01061  7.00223E-04 0.01146  1.59510E-03 0.00856  3.73465E-04 0.01695  8.23799E-05 0.03488 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10626E-01 0.01427  1.24468E-02 0.00251  3.22161E-02 0.00014  1.06760E-01 0.00055  3.01124E-01 0.00030  1.26041E+00 0.00086  5.89037E+00 0.02795 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51648E-03 0.00780  2.49385E-04 0.02861  8.11013E-04 0.01576  6.29205E-04 0.01701  1.43041E-03 0.01197  3.20048E-04 0.02535  7.64145E-05 0.05350 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.14096E-01 0.02223  1.24782E-02 8.4E-05  3.22194E-02 0.00021  1.06780E-01 0.00077  3.01021E-01 0.00041  1.25922E+00 0.00117  6.88270E+00 0.02083 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68112E-07 0.00137  8.67591E-07 0.00137  1.01524E-06 0.01670 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.84236E-07 0.00128  7.83765E-07 0.00129  9.16910E-07 0.01663 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48738E-03 0.00945  2.51343E-04 0.03281  8.24017E-04 0.01882  6.19359E-04 0.02086  1.39360E-03 0.01467  3.27798E-04 0.02882  7.12594E-05 0.06151 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07735E-01 0.02415  1.24788E-02 0.00012  3.22161E-02 0.00028  1.06861E-01 0.00111  3.01252E-01 0.00054  1.25775E+00 0.00178  6.95982E+00 0.02960 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.64780E-07 0.01721  7.64133E-07 0.01722  9.02917E-07 0.05245 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.91109E-07 0.01721  6.90522E-07 0.01722  8.17017E-07 0.05255 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09337E-03 0.04358  2.12458E-04 0.12565  7.35402E-04 0.07451  5.42926E-04 0.09267  1.30256E-03 0.06400  2.50078E-04 0.14095  4.99462E-05 0.28490 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.73136E-01 0.08956  1.24784E-02 0.00011  3.22502E-02 0.00082  1.06884E-01 0.00306  3.00932E-01 0.00193  1.26160E+00 0.00391  7.65699E+00 0.08027 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08641E-03 0.04279  2.13017E-04 0.12397  7.46062E-04 0.07229  5.43349E-04 0.09187  1.28359E-03 0.06319  2.48053E-04 0.13658  5.23481E-05 0.28162 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.74846E-01 0.08790  1.24784E-02 0.00011  3.22529E-02 0.00080  1.06930E-01 0.00309  3.00985E-01 0.00194  1.26143E+00 0.00390  7.65699E+00 0.08027 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.07838E+03 0.04056 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.58951E-07 0.00085 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.75950E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.51453E-03 0.00581 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.09280E+03 0.00587 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94428E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74452E-05 0.00823  6.74506E-05 0.00824  1.09187E-06 0.41382 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.61457E-05 0.02406  5.61705E-05 0.02406  8.88286E-07 0.54946 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.05706E-04 0.02067  2.05920E-04 0.02065  1.43346E-04 0.40599 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60981E+01 0.01215 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47057E+01 0.00027  3.93296E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Apr 18 01:04:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01794E+00  1.00413E+00  1.00059E+00  9.95785E-01  9.84183E-01  9.94452E-01  9.96785E-01  9.93133E-01  9.91454E-01  1.00046E+00  1.00346E+00  9.91643E-01  1.00937E+00  1.01397E+00  1.00570E+00  1.00762E+00  9.89880E-01  9.99430E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.62390E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73761E-01 9.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36287E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50392E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60246E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47535E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46590E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27962E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.42073E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500577 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50074E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50074E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.93002E+03 ;
RUNNING_TIME              (idx, 1)        =  4.21819E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.19358E+00  9.68000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.19163E+02  3.44692E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.94150E-01  1.40667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.96709E+01  1.25752E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.20607E+02  4.89964E+02 ];
CPU_USAGE                 (idx, 1)        = 9.31684 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98651E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.25883E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 190 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.92125E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74135E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.53284E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92778E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34059E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95918E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57319E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53967E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39619E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36482E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14487E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69322E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.24042E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58253E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34791E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31798E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53034E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99694E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59943E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23980E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03801E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.16574E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.02938E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04965E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06777E-01 -2.35760E+28  2.44372E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02142E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.54281E+18 0.00416  1.64699E-02 0.00407 ];
U233_FISS                 (idx, [1:   4]) = [  7.94303E+19 0.00059  8.48024E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.12416E+18 0.00209  7.60600E-02 0.00203 ];
U238_FISS                 (idx, [1:   4]) = [  1.57862E+14 0.40571  1.69548E-06 0.40572 ];
PU239_FISS                (idx, [1:   4]) = [  6.12570E+17 0.00656  6.53992E-03 0.00654 ];
PU240_FISS                (idx, [1:   4]) = [  7.23386E+16 0.01910  7.72637E-04 0.01914 ];
PU241_FISS                (idx, [1:   4]) = [  1.26793E+17 0.01433  1.35401E-03 0.01435 ];
TH232_CAPT                (idx, [1:   4]) = [  9.31253E+19 0.00056  6.45764E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.16053E+19 0.00147  8.04770E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49397E+18 0.00326  1.72944E-02 0.00323 ];
U238_CAPT                 (idx, [1:   4]) = [  2.85878E+15 0.09154  1.98545E-05 0.09157 ];
PU239_CAPT                (idx, [1:   4]) = [  2.89642E+17 0.00954  2.00837E-03 0.00951 ];
PU240_CAPT                (idx, [1:   4]) = [  1.91566E+17 0.01143  1.32835E-03 0.01142 ];
PU241_CAPT                (idx, [1:   4]) = [  2.90312E+16 0.02957  2.01323E-04 0.02956 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83492E+17 0.01229  1.27225E-03 0.01226 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002949 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08129E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002949 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5491063 5.49558E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3566803 3.56944E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 945083 9.45793E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002949 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.32248E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36596E+20 5.1E-06  2.36596E+20 5.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35958E+19 1.1E-07  9.35958E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44182E+20 0.00029  1.28994E+20 0.00020  1.51887E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37778E+20 0.00018  2.22590E+20 0.00012  1.51887E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62411E+20 0.00035  2.62411E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.51098E+22 0.00025  3.33218E+22 0.00022  1.78792E+21 0.00191 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.48195E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62598E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17310E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.01216E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.01216E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.21595E-01 0.04267 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.20867E-01 0.01085 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.70973E-04 0.02035 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.27536E+03 0.02598 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.05426E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.10566E-01 0.03181 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.43431E-01 0.03181 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52784E+00 5.0E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00057E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.01956E-01 0.00046  5.62006E-02 0.00045  1.92141E-04 0.00935 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.01974E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.01665E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.01974E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96232E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84603E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84704E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59624E-02 0.00150 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59419E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08431E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07869E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94355E-03 0.00570  2.82738E-04 0.01851  9.27941E-04 0.01094  7.18146E-04 0.01231  1.57309E-03 0.00909  3.60449E-04 0.01760  8.11851E-05 0.03975 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05041E-01 0.01526  1.24805E-02 0.00013  3.21936E-02 0.00017  1.06752E-01 0.00057  3.01339E-01 0.00031  1.25775E+00 0.00106  5.72695E+00 0.03051 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47261E-03 0.00801  2.40509E-04 0.02721  8.19719E-04 0.01649  6.30089E-04 0.01833  1.39358E-03 0.01237  3.17497E-04 0.02584  7.12088E-05 0.05486 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.03820E-01 0.02219  1.24835E-02 0.00021  3.21931E-02 0.00021  1.06698E-01 0.00076  3.01581E-01 0.00044  1.25804E+00 0.00137  6.88371E+00 0.02160 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.63554E-07 0.00148  8.63116E-07 0.00148  9.89034E-07 0.01674 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.78821E-07 0.00140  7.78425E-07 0.00140  8.92026E-07 0.01674 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40825E-03 0.00940  2.34465E-04 0.03565  8.01111E-04 0.01889  6.29126E-04 0.01999  1.37480E-03 0.01476  3.00767E-04 0.03156  6.79883E-05 0.06770 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07011E-01 0.02652  1.24788E-02 0.00014  3.22109E-02 0.00026  1.06907E-01 0.00114  3.01639E-01 0.00053  1.25899E+00 0.00165  7.29075E+00 0.02877 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.54580E-07 0.01708  7.53993E-07 0.01708  8.57480E-07 0.05134 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.80815E-07 0.01708  6.80287E-07 0.01708  7.73733E-07 0.05134 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92391E-03 0.04545  2.54696E-04 0.13309  7.35291E-04 0.07633  4.70930E-04 0.09678  1.23461E-03 0.06535  1.98625E-04 0.14667  2.97524E-05 0.26542 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.48147E-01 0.10443  1.24771E-02 0.00013  3.21980E-02 0.00089  1.06663E-01 0.00315  3.01302E-01 0.00196  1.24866E+00 0.00752  7.38841E+00 0.08398 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91114E-03 0.04443  2.53724E-04 0.12882  7.29045E-04 0.07456  4.70813E-04 0.09366  1.23167E-03 0.06388  1.94163E-04 0.14027  3.17287E-05 0.27827 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.44475E-01 0.09823  1.24770E-02 0.00014  3.21964E-02 0.00090  1.06644E-01 0.00309  3.01260E-01 0.00195  1.24844E+00 0.00751  7.40311E+00 0.08350 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88873E+03 0.04239 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.55136E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.71232E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38114E-03 0.00703 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95385E+03 0.00698 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.08045E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75447E-05 0.00702  6.75193E-05 0.00703  1.90364E-06 0.32611 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.67738E-05 0.01949  5.67120E-05 0.01950  1.53719E-06 0.38956 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.62789E-04 0.02023  2.62721E-04 0.02021  2.87135E-04 0.31943 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61975E+01 0.01167 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46590E+01 0.00028  3.90114E+01 0.00041 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Apr 18 01:39:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02466E+00  1.00451E+00  1.00592E+00  9.98996E-01  9.87719E-01  1.00380E+00  9.99506E-01  9.98689E-01  1.00070E+00  9.98597E-01  9.98950E-01  9.98023E-01  9.94024E-01  9.94240E-01  9.95546E-01  9.96069E-01  1.00045E+00  9.99597E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.66251E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73375E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34728E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49042E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61647E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48129E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47177E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33987E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.62260E-01 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500483 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50062E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50062E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.26489E+03 ;
RUNNING_TIME              (idx, 1)        =  4.56548E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.29310E+00  9.95167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.53734E+02  3.45710E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.09317E-01  1.51667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.09101E+01  1.23915E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.55357E+02  4.89941E+02 ];
CPU_USAGE                 (idx, 1)        = 9.34160 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98717E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.28644E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 190 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.91595E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73917E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.40233E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91510E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33399E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96650E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57160E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64087E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56989E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45846E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16215E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76517E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39615E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93272E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33979E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31200E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52198E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01470E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00939E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22795E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02626E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.36298E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00724E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.05018E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93016E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61342E-01 -3.56238E+28  2.56420E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01651E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.51520E+18 0.00432  1.61861E-02 0.00429 ];
U233_FISS                 (idx, [1:   4]) = [  7.92435E+19 0.00055  8.46520E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.10516E+18 0.00203  7.59004E-02 0.00196 ];
U238_FISS                 (idx, [1:   4]) = [  2.64928E+14 0.31267  2.82058E-06 0.31268 ];
PU239_FISS                (idx, [1:   4]) = [  6.52798E+17 0.00627  6.97330E-03 0.00624 ];
PU240_FISS                (idx, [1:   4]) = [  8.51191E+16 0.01668  9.09446E-04 0.01670 ];
PU241_FISS                (idx, [1:   4]) = [  1.54808E+17 0.01364  1.65401E-03 0.01367 ];
TH232_CAPT                (idx, [1:   4]) = [  9.23553E+19 0.00052  6.40422E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15506E+19 0.00149  8.00969E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49307E+18 0.00332  1.72887E-02 0.00334 ];
U238_CAPT                 (idx, [1:   4]) = [  2.57558E+15 0.10530  1.78629E-05 0.10525 ];
PU239_CAPT                (idx, [1:   4]) = [  3.10754E+17 0.00962  2.15474E-03 0.00961 ];
PU240_CAPT                (idx, [1:   4]) = [  2.25015E+17 0.01106  1.56046E-03 0.01107 ];
PU241_CAPT                (idx, [1:   4]) = [  3.35875E+16 0.02970  2.32830E-04 0.02967 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90203E+17 0.01147  1.31890E-03 0.01146 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002499 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08154E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002499 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487992 5.49284E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3562776 3.56555E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 951731 9.52417E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002499 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.08616E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36661E+20 4.7E-06  2.36661E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35895E+19 1.1E-07  9.35895E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44218E+20 0.00027  1.28569E+20 0.00019  1.56488E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37807E+20 0.00016  2.22159E+20 0.00011  1.56488E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62545E+20 0.00035  2.62545E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54031E+22 0.00026  3.35868E+22 0.00024  1.81630E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50065E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62814E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17525E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  7.81548E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.81548E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.03623E-01 0.03916 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.15790E-01 0.00923 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.46089E-04 0.01871 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.98557E+03 0.02585 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.04764E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.85087E-01 0.02601 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.10453E-01 0.02601 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52871E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.01672E-01 0.00043  5.61562E-02 0.00042  1.96399E-04 0.00884 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.01475E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.01455E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.01475E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96378E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84741E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84863E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59391E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59165E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08829E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.08703E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.98438E-03 0.00544  2.79525E-04 0.02055  9.25383E-04 0.01092  7.16157E-04 0.01267  1.60561E-03 0.00868  3.74388E-04 0.01811  8.33124E-05 0.03534 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.15438E-01 0.01348  1.24792E-02 7.5E-05  3.21954E-02 0.00016  1.06703E-01 0.00056  3.01291E-01 0.00029  1.25711E+00 0.00109  6.10107E+00 0.02752 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49689E-03 0.00790  2.34954E-04 0.03044  8.07087E-04 0.01658  6.17597E-04 0.01775  1.43442E-03 0.01239  3.31089E-04 0.02551  7.17506E-05 0.05565 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.16750E-01 0.02096  1.24787E-02 5.9E-05  3.21935E-02 0.00023  1.06642E-01 0.00068  3.01371E-01 0.00041  1.25704E+00 0.00138  7.25780E+00 0.01905 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.74049E-07 0.00145  8.73525E-07 0.00145  1.02619E-06 0.02192 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.88051E-07 0.00139  7.87579E-07 0.00139  9.25216E-07 0.02185 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.49366E-03 0.00898  2.37496E-04 0.03375  8.04542E-04 0.01796  6.28699E-04 0.02069  1.43143E-03 0.01433  3.16535E-04 0.03103  7.49570E-05 0.05936 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.18428E-01 0.02516  1.24788E-02 0.00011  3.21909E-02 0.00029  1.06833E-01 0.00108  3.01241E-01 0.00048  1.25835E+00 0.00203  7.04485E+00 0.02887 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.60645E-07 0.01723  7.59938E-07 0.01723  9.09837E-07 0.04966 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.85930E-07 0.01722  6.85295E-07 0.01723  8.19782E-07 0.04962 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14045E-03 0.04273  2.16393E-04 0.15114  7.76048E-04 0.07741  4.67198E-04 0.09052  1.30442E-03 0.06469  2.88325E-04 0.12989  8.80610E-05 0.22598 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.63026E-01 0.11519  1.24779E-02 0.00012  3.21442E-02 0.00085  1.06763E-01 0.00337  3.01048E-01 0.00190  1.24891E+00 0.00594  7.68231E+00 0.07582 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.11327E-03 0.04171  2.15714E-04 0.14489  7.74062E-04 0.07732  4.47177E-04 0.09030  1.29266E-03 0.06400  2.92224E-04 0.12861  9.14315E-05 0.22618 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.65071E-01 0.11258  1.24779E-02 0.00012  3.21491E-02 0.00085  1.06770E-01 0.00336  3.00999E-01 0.00188  1.24857E+00 0.00599  7.69216E+00 0.07562 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.16818E+03 0.03988 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.62803E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.77907E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47441E-03 0.00586 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.02760E+03 0.00590 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.26195E-09 0.00102 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.90925E-05 0.00536  6.91061E-05 0.00537  1.57311E-06 0.33548 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.06457E-05 0.01671  6.06724E-05 0.01669  1.07365E-06 0.43236 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.31552E-04 0.01800  3.31874E-04 0.01788  2.49599E-04 0.31460 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58069E+01 0.01114 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47177E+01 0.00027  3.90197E+01 0.00036 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 24])  = 'msfr_mix1_benchmark_burn' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Apr 17 18:02:29 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Apr 18 01:57:43 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650229349159 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 4 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03410E+00  1.00140E+00  1.00468E+00  9.84326E-01  9.90970E-01  9.96863E-01  9.99666E-01  1.00209E+00  1.00700E+00  1.00247E+00  9.95857E-01  9.88860E-01  9.92061E-01  9.99659E-01  1.00209E+00  1.00310E+00  9.98803E-01  9.96007E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.70823E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72918E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.33280E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.47844E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62714E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48922E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47961E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.39866E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.84018E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500563 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50047E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50047E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.44672E+03 ;
RUNNING_TIME              (idx, 1)        =  4.75233E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87983E-01  9.87983E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.39213E+00  9.90333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.72258E+02  1.85240E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.24150E-01  1.48333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.13311E+01  4.20917E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.74860E+02  4.74860E+02 ];
CPU_USAGE                 (idx, 1)        = 9.35693 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97955E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.30400E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 190 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.91112E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73859E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.62765E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90686E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32896E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96983E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57143E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65592E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61737E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47125E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16269E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78633E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44280E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03320E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33834E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31097E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52050E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01706E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13609E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22159E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02322E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.40330E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99167E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.05206E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57344E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16367E-01 -4.77730E+28  2.68569E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01344E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.52028E+18 0.00431  1.62430E-02 0.00429 ];
U233_FISS                 (idx, [1:   4]) = [  7.92446E+19 0.00057  8.46649E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  7.06336E+18 0.00184  7.54659E-02 0.00178 ];
U238_FISS                 (idx, [1:   4]) = [  1.84399E+14 0.37514  1.97461E-06 0.37517 ];
PU239_FISS                (idx, [1:   4]) = [  6.59622E+17 0.00628  7.04728E-03 0.00625 ];
PU240_FISS                (idx, [1:   4]) = [  8.87332E+16 0.01724  9.47669E-04 0.01715 ];
PU241_FISS                (idx, [1:   4]) = [  1.56133E+17 0.01234  1.66827E-03 0.01234 ];
TH232_CAPT                (idx, [1:   4]) = [  9.20172E+19 0.00052  6.37301E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15255E+19 0.00155  7.98232E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50036E+18 0.00309  1.73170E-02 0.00305 ];
U238_CAPT                 (idx, [1:   4]) = [  2.94486E+15 0.09324  2.03957E-05 0.09327 ];
PU239_CAPT                (idx, [1:   4]) = [  3.14406E+17 0.00866  2.17760E-03 0.00866 ];
PU240_CAPT                (idx, [1:   4]) = [  2.30279E+17 0.01120  1.59493E-03 0.01121 ];
PU241_CAPT                (idx, [1:   4]) = [  3.47214E+16 0.02760  2.40441E-04 0.02757 ];
SM149_CAPT                (idx, [1:   4]) = [  1.88262E+17 0.01177  1.30397E-03 0.01179 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001875 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08113E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001875 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5484596 5.48970E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3555632 3.55867E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 961647 9.62447E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001875 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.04891E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36672E+20 4.8E-06  2.36672E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35888E+19 1.0E-07  9.35888E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44405E+20 0.00027  1.28278E+20 0.00018  1.61266E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37993E+20 0.00016  2.21867E+20 0.00010  1.61266E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.63015E+20 0.00034  2.63015E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.57478E+22 0.00025  3.39067E+22 0.00023  1.84104E+21 0.00183 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.53152E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.63309E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17940E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  7.64805E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.64805E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.01855E-01 0.03442 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07323E-01 0.00853 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.19118E-04 0.01645 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.53718E+03 0.02720 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.03763E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 9.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.41367E-01 0.02145 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.60390E-01 0.02145 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52885E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.99743E-01 0.00044  5.60507E-02 0.00043  1.94231E-04 0.00959 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.99827E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.99882E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.99827E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95651E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84961E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84993E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59045E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58958E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08378E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.08885E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96422E-03 0.00552  2.77597E-04 0.01958  9.30454E-04 0.01134  7.19309E-04 0.01281  1.59947E-03 0.00858  3.55927E-04 0.01742  8.14601E-05 0.03757 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05333E-01 0.01512  1.24807E-02 0.00014  3.21929E-02 0.00016  1.06706E-01 0.00048  3.01433E-01 0.00029  1.25728E+00 0.00108  5.83061E+00 0.02825 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46261E-03 0.00794  2.38259E-04 0.03002  8.11808E-04 0.01574  6.37417E-04 0.01774  1.39876E-03 0.01205  3.06348E-04 0.02611  7.00215E-05 0.05105 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02711E-01 0.02052  1.24807E-02 0.00022  3.21973E-02 0.00021  1.06625E-01 0.00066  3.01420E-01 0.00042  1.25834E+00 0.00153  6.86160E+00 0.02072 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.87918E-07 0.00155  8.87576E-07 0.00156  9.84190E-07 0.01580 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.98834E-07 0.00148  7.98526E-07 0.00149  8.85501E-07 0.01579 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45576E-03 0.00967  2.44572E-04 0.03393  8.11951E-04 0.01984  6.40499E-04 0.02152  1.38611E-03 0.01495  3.02003E-04 0.02991  7.06209E-05 0.06070 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.00513E-01 0.02393  1.24822E-02 0.00021  3.22039E-02 0.00028  1.06700E-01 0.00096  3.01492E-01 0.00053  1.25996E+00 0.00182  7.03139E+00 0.03033 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.76407E-07 0.01767  7.75981E-07 0.01768  8.31852E-07 0.05860 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.98758E-07 0.01769  6.98375E-07 0.01770  7.48678E-07 0.05861 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08141E-03 0.04365  2.48748E-04 0.15169  6.51429E-04 0.08947  5.93550E-04 0.08595  1.27443E-03 0.06909  2.56211E-04 0.12838  5.70385E-05 0.28601 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.71653E-01 0.09180  1.24901E-02 0.00112  3.22601E-02 0.00085  1.06752E-01 0.00319  3.01890E-01 0.00199  1.25950E+00 0.00612  5.26319E+00 0.13706 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09576E-03 0.04248  2.50506E-04 0.14749  6.63888E-04 0.08736  5.81310E-04 0.08515  1.28693E-03 0.06760  2.56591E-04 0.12519  5.65306E-05 0.28018 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.62074E-01 0.08450  1.24901E-02 0.00112  3.22613E-02 0.00085  1.06788E-01 0.00321  3.01955E-01 0.00198  1.25984E+00 0.00600  5.26319E+00 0.13706 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.01379E+03 0.04055 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.77123E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.89121E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40081E-03 0.00587 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.87867E+03 0.00598 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.45395E-09 0.00110 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.88496E-05 0.00482  6.88506E-05 0.00482  1.59361E-06 0.33993 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.28137E-05 0.01614  6.27873E-05 0.01616  1.61751E-06 0.50401 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.01560E-04 0.01598  4.02257E-04 0.01591  2.35227E-04 0.33150 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60581E+01 0.01069 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47961E+01 0.00028  3.90336E+01 0.00039 ];

