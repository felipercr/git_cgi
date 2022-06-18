
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 15:50:39 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01587E+00  9.99996E-01  9.93569E-01  1.00145E+00  1.00969E+00  1.00492E+00  1.01741E+00  9.94228E-01  1.00118E+00  1.00064E+00  9.97089E-01  9.94790E-01  1.00233E+00  9.83967E-01  9.89963E-01  9.92419E-01  9.94633E-01  1.00584E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.68450E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63155E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39461E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53334E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64582E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46879E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45971E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16214E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16297E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500135 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50029E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50029E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.18660E+02 ;
RUNNING_TIME              (idx, 1)        =  2.29444E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.84083E-01  1.84083E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.16786E+01  2.16786E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.02350E-01  4.86167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.29390E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.53004 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97637E+00 3.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.51995E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  2.79890E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.20110E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13464E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.79890E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.20110E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.69009E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.42800E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.69009E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.42800E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29728E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.13367E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.31768E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.46200E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.68648E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.53280E+18 0.00403  1.63464E-02 0.00398 ];
U233_FISS                 (idx, [1:   4]) = [  9.22355E+19 0.00051  9.83654E-01 6.6E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.17655E+19 0.00055  7.54844E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34077E+19 0.00134  1.10291E-01 0.00131 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001174 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15445E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001174 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5133843 5.13927E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3959887 3.96402E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907444 9.08250E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001174 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.02914E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36437E+20 3.6E-06  2.36437E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.2E-07  9.37766E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21567E+20 0.00032  1.08272E+20 0.00023  1.32951E+19 0.00196 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15344E+20 0.00018  2.02049E+20 0.00012  1.32951E+19 0.00196 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36550E+20 0.00035  2.36550E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.13077E+22 0.00026  2.96896E+22 0.00023  1.61804E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.14851E+19 0.00134 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.36829E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05611E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.26826E-01 0.07679 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.90453E-01 0.00844 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.93201E-04 0.01513 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.88492E+03 0.02024 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09185E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.01250E-01 0.06604 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.64947E-01 0.06604 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52128E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.99553E-01 0.00041  6.22657E-02 0.00040  1.99806E-04 0.00869 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.99512E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.99574E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.99512E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09936E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04941E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04912E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30230E-02 0.00127 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30250E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.46503E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46467E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.21032E-03 0.00583  2.66008E-04 0.02072  7.89256E-04 0.01143  5.93922E-04 0.01257  1.24841E-03 0.00901  2.62962E-04 0.01996  4.97582E-05 0.04573 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40826E-01 0.01506  1.24440E-02 0.00251  3.23345E-02 9.6E-05  1.05976E-01 0.00058  2.97342E-01 0.00027  1.23592E+00 0.00022  4.47912E+00 0.04217 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.18882E-03 0.00842  2.65180E-04 0.03179  7.90738E-04 0.01691  5.91764E-04 0.01897  1.23511E-03 0.01277  2.59255E-04 0.02819  4.67789E-05 0.06963 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.37641E-01 0.02194  1.24753E-02 4.1E-05  3.23385E-02 0.00014  1.06001E-01 0.00083  2.97314E-01 0.00039  1.23576E+00 0.00031  6.18493E+00 0.02977 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.35760E-07 0.00118  8.35526E-07 0.00119  9.10860E-07 0.01551 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.35335E-07 0.00112  8.35101E-07 0.00113  9.10554E-07 0.01555 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20302E-03 0.00879  2.78023E-04 0.03138  7.94026E-04 0.01798  5.87336E-04 0.01863  1.24002E-03 0.01427  2.55778E-04 0.03068  4.78381E-05 0.07059 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.33222E-01 0.02370  1.24751E-02 5.6E-05  3.23379E-02 0.00017  1.05995E-01 0.00101  2.97266E-01 0.00043  1.23584E+00 0.00037  6.28339E+00 0.04208 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.44382E-07 0.01706  7.44098E-07 0.01706  7.93517E-07 0.05644 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.44134E-07 0.01705  7.43848E-07 0.01705  7.93354E-07 0.05647 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.70464E-03 0.03957  2.15219E-04 0.14408  6.63281E-04 0.08021  5.48325E-04 0.08400  1.00469E-03 0.06039  2.41759E-04 0.13511  3.13606E-05 0.32498 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.73782E-01 0.12896  1.24752E-02 0.00014  3.23395E-02 0.00051  1.06277E-01 0.00322  2.96763E-01 0.00137  1.23645E+00 0.00108  6.75662E+00 0.14230 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.70669E-03 0.03956  2.11652E-04 0.13293  6.75515E-04 0.08008  5.48107E-04 0.08228  1.00193E-03 0.05860  2.36971E-04 0.13556  3.25090E-05 0.32446 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.71181E-01 0.12952  1.24754E-02 0.00013  3.23412E-02 0.00051  1.06298E-01 0.00323  2.96679E-01 0.00134  1.23654E+00 0.00107  6.75661E+00 0.14230 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.64802E+03 0.03610 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.30072E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.29642E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20902E-03 0.00560 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.86621E+03 0.00557 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.23649E-09 0.00141 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.17931E-05 0.00417  7.18285E-05 0.00425  1.93407E-06 0.30538 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.72544E-05 0.01651  7.73198E-05 0.01653  1.93755E-06 0.43651 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.74815E-04 0.01467  4.75236E-04 0.01468  3.60614E-04 0.30026 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73285E+01 0.01197 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45971E+01 0.00027  4.04447E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 16:24:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00684E+00  1.00435E+00  9.95504E-01  9.98952E-01  9.98776E-01  1.00270E+00  9.94909E-01  1.00064E+00  9.90794E-01  9.93603E-01  1.00223E+00  1.00682E+00  1.00010E+00  1.00126E+00  1.00136E+00  9.96033E-01  9.94779E-01  1.01035E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.68098E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63190E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39398E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53259E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64924E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47637E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46733E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17880E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16467E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500586 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50035E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50035E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.56236E+02 ;
RUNNING_TIME              (idx, 1)        =  5.68690E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.10100E-01  1.26017E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.54531E+01  3.37744E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.84000E-02  1.84000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.52717E-01  5.03167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.68635E+01  3.40563E+02 ];
CPU_USAGE                 (idx, 1)        = 9.78101 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98318E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.78364E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  5.76414E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60312E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65457E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.46881E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.03466E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26470E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46895E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.44804E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65315E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.38490E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.75027E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06313E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17812E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44279E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35671E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17525E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55419E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59266E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38167E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.38931E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51286E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50526E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.28076E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.55423E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08113E-04 -2.38711E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.79813E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.54542E+18 0.00382  1.64825E-02 0.00377 ];
U233_FISS                 (idx, [1:   4]) = [  9.21855E+19 0.00051  9.83216E-01 6.4E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.39049E+14 0.31266  2.54750E-06 0.31265 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28769E+19 0.00052  7.50568E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34486E+19 0.00133  1.08683E-01 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  1.19468E+14 0.44500  9.56806E-07 0.44503 ];
SM149_CAPT                (idx, [1:   4]) = [  1.29163E+16 0.04307  1.04367E-04 0.04310 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001382 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14773E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001382 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5175285 5.18067E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3921467 3.92536E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904630 9.05448E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001382 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.75671E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36430E+20 3.5E-06  2.36430E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37763E+19 1.2E-07  9.37763E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23743E+20 0.00030  1.10316E+20 0.00022  1.34275E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17519E+20 0.00017  2.04092E+20 0.00012  1.34275E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38856E+20 0.00037  2.38856E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.16557E+22 0.00025  3.00206E+22 0.00024  1.63507E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.16276E+19 0.00136 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39147E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06820E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.62530E-01 0.07097 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.80104E-01 0.00874 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.45133E-04 0.01532 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.46116E+03 0.02223 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09464E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 9.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.32611E-01 0.06164 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.93432E-01 0.06164 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52121E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.89662E-01 0.00039  6.16516E-02 0.00038  2.02990E-04 0.00846 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.89794E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.89897E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.89794E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08833E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05343E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05057E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29711E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30063E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.46850E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46906E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30142E-03 0.00562  2.67439E-04 0.01975  8.11807E-04 0.01112  6.03063E-04 0.01311  1.28323E-03 0.00924  2.84541E-04 0.01796  5.13366E-05 0.04516 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50782E-01 0.01491  1.24442E-02 0.00251  3.23371E-02 9.0E-05  1.06094E-01 0.00062  2.97362E-01 0.00027  1.23559E+00 0.00023  4.65500E+00 0.04065 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25852E-03 0.00865  2.62928E-04 0.02969  7.99141E-04 0.01680  5.96004E-04 0.01959  1.26500E-03 0.01345  2.81360E-04 0.02855  5.40869E-05 0.06656 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56030E-01 0.02213  1.24756E-02 3.9E-05  3.23422E-02 0.00014  1.06067E-01 0.00080  2.97405E-01 0.00038  1.23590E+00 0.00029  6.45478E+00 0.02820 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.45151E-07 0.00120  8.44794E-07 0.00120  9.50010E-07 0.01775 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.36359E-07 0.00113  8.36006E-07 0.00113  9.40057E-07 0.01770 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27574E-03 0.00857  2.69618E-04 0.03027  8.16478E-04 0.01754  6.05684E-04 0.02077  1.25181E-03 0.01393  2.80623E-04 0.02910  5.15259E-05 0.06863 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51174E-01 0.02383  1.24758E-02 4.8E-05  3.23337E-02 0.00015  1.06158E-01 0.00107  2.97329E-01 0.00044  1.23569E+00 0.00038  6.52108E+00 0.03934 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.50294E-07 0.01704  7.49987E-07 0.01704  7.82457E-07 0.05454 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.42664E-07 0.01704  7.42360E-07 0.01704  7.74554E-07 0.05455 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.90833E-03 0.04157  2.07022E-04 0.14302  7.68387E-04 0.07866  6.10371E-04 0.08855  1.00485E-03 0.06305  2.71338E-04 0.13629  4.63653E-05 0.30496 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.77614E-01 0.10541  1.24752E-02 0.00015  3.23257E-02 0.00044  1.06604E-01 0.00334  2.96758E-01 0.00137  1.23512E+00 0.00109  7.58200E+00 0.09930 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90917E-03 0.04098  2.06609E-04 0.13859  7.73282E-04 0.07671  5.95410E-04 0.08655  1.01909E-03 0.06241  2.71128E-04 0.13127  4.36462E-05 0.31809 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.70534E-01 0.09889  1.24753E-02 0.00015  3.23249E-02 0.00044  1.06655E-01 0.00338  2.96850E-01 0.00138  1.23514E+00 0.00110  7.58200E+00 0.09930 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.90784E+03 0.03828 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.39211E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.30482E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29705E-03 0.00652 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.92867E+03 0.00649 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.18601E-09 0.00122 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.21857E-05 0.00464  7.21866E-05 0.00465  1.92582E-06 0.30495 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.64935E-05 0.01631  7.65769E-05 0.01635  1.47937E-06 0.37651 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.28445E-04 0.01458  4.28660E-04 0.01454  3.60227E-04 0.31010 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73837E+01 0.01167 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46733E+01 0.00025  4.05212E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 17:00:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00544E+00  9.97451E-01  1.00571E+00  1.00114E+00  9.94380E-01  1.00236E+00  1.00560E+00  9.97601E-01  9.95778E-01  9.89009E-01  9.94406E-01  9.96216E-01  1.00602E+00  1.00266E+00  1.00061E+00  1.00979E+00  9.94550E-01  1.00127E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.66084E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63392E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39654E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53447E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64119E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48787E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47881E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19306E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.15458E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499887 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50023E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50023E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  9.13072E+02 ;
RUNNING_TIME              (idx, 1)        =  9.27257E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.41783E-01  1.31683E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  9.11493E+01  3.56962E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.75333E-02  1.91333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.03500E-01  5.07333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  9.27201E+01  4.99823E+02 ];
CPU_USAGE                 (idx, 1)        = 9.84702 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98385E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.85294E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.38129E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65353E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.23236E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85332E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29541E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49704E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49299E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.74785E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39696E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.85058E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.95989E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89706E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60097E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49557E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34609E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33331E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54292E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.66152E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17894E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.00456E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51083E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.32266E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.49866E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.68269E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.75710E-04 -1.27115E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.96561E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.58039E+18 0.00395  1.68460E-02 0.00388 ];
U233_FISS                 (idx, [1:   4]) = [  9.20998E+19 0.00048  9.81782E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.06286E+16 0.05158  1.13366E-04 0.05160 ];
TH232_CAPT                (idx, [1:   4]) = [  9.42943E+19 0.00052  7.44828E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34277E+19 0.00124  1.06068E-01 0.00122 ];
U235_CAPT                 (idx, [1:   4]) = [  3.79831E+15 0.08501  3.00038E-05 0.08497 ];
SM149_CAPT                (idx, [1:   4]) = [  6.08238E+16 0.01972  4.80584E-04 0.01976 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000917 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16825E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000917 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5224211 5.22992E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3871205 3.87536E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905501 9.06399E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000917 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.87430E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36422E+20 3.7E-06  2.36422E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37758E+19 1.2E-07  9.37758E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26569E+20 0.00030  1.12912E+20 0.00022  1.36572E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20345E+20 0.00017  2.06688E+20 0.00012  1.36572E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42067E+20 0.00034  2.42067E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.21115E+22 0.00025  3.04505E+22 0.00023  1.66104E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.19418E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42287E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08537E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51431E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51431E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.43892E-01 0.05399 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23769E-01 0.01011 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.78170E-04 0.02093 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.49376E+03 0.02421 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09365E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.28417E-01 0.04217 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.71479E-01 0.04217 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52114E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77088E-01 0.00042  6.08665E-02 0.00041  1.97383E-04 0.00932 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.76942E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.76723E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.76942E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07435E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05703E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05651E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29242E-02 0.00127 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29290E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48408E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48075E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34450E-03 0.00555  2.74764E-04 0.02002  8.28771E-04 0.01148  6.14002E-04 0.01256  1.29409E-03 0.00885  2.81849E-04 0.01920  5.10247E-05 0.04512 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42499E-01 0.01535  1.24443E-02 0.00251  3.23363E-02 9.3E-05  1.05946E-01 0.00053  2.97360E-01 0.00027  1.23544E+00 0.00023  4.38031E+00 0.04282 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27675E-03 0.00809  2.63372E-04 0.02959  8.16878E-04 0.01735  6.06953E-04 0.01907  1.25822E-03 0.01284  2.81416E-04 0.02864  4.99150E-05 0.06875 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48011E-01 0.02341  1.24754E-02 3.7E-05  3.23370E-02 0.00013  1.05843E-01 0.00066  2.97282E-01 0.00038  1.23531E+00 0.00031  6.23827E+00 0.02966 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.57904E-07 0.00118  8.57594E-07 0.00118  9.58015E-07 0.01643 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.38192E-07 0.00111  8.37888E-07 0.00111  9.35969E-07 0.01637 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24094E-03 0.00964  2.75664E-04 0.03088  7.87943E-04 0.01882  5.88453E-04 0.02045  1.27498E-03 0.01481  2.67890E-04 0.03214  4.60116E-05 0.07853 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.35596E-01 0.02505  1.24759E-02 4.8E-05  3.23327E-02 0.00015  1.05885E-01 0.00094  2.97378E-01 0.00043  1.23595E+00 0.00040  6.04726E+00 0.04462 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.61390E-07 0.01711  7.61031E-07 0.01711  8.01034E-07 0.05505 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.44133E-07 0.01710  7.43783E-07 0.01711  7.82770E-07 0.05498 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85897E-03 0.04297  2.36038E-04 0.13309  7.39173E-04 0.07726  4.78615E-04 0.09288  1.13004E-03 0.06293  2.45864E-04 0.12587  2.92374E-05 0.35611 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.34170E-01 0.08505  1.24781E-02 7.6E-05  3.23373E-02 0.00050  1.05555E-01 0.00259  2.97933E-01 0.00169  1.23368E+00 0.00122  6.44147E+00 0.16948 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87011E-03 0.04237  2.44284E-04 0.12981  7.37375E-04 0.07675  4.88891E-04 0.09217  1.12423E-03 0.06299  2.44833E-04 0.12512  3.04887E-05 0.35328 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.27638E-01 0.08161  1.24782E-02 7.3E-05  3.23375E-02 0.00050  1.05573E-01 0.00262  2.97987E-01 0.00168  1.23377E+00 0.00121  6.44147E+00 0.16948 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79462E+03 0.04000 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.54665E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.35026E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26961E-03 0.00642 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82570E+03 0.00641 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99764E-09 0.00098 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.96779E-05 0.00585  6.96901E-05 0.00585  2.72586E-07 0.70979 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.80974E-05 0.02167  6.80479E-05 0.02173  3.61109E-07 0.74010 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.69241E-04 0.02055  2.69946E-04 0.02058  6.24656E-05 0.70758 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75684E+01 0.01138 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47881E+01 0.00027  4.06530E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 17:36:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01553E+00  9.95142E-01  1.00314E+00  1.00205E+00  9.98264E-01  9.95802E-01  1.00158E+00  9.95887E-01  1.00715E+00  1.00634E+00  1.00207E+00  9.91263E-01  9.91752E-01  1.00414E+00  9.93183E-01  9.96579E-01  1.00529E+00  9.94861E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.64754E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63525E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39446E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53202E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63643E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49615E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48708E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.21697E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.15022E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500563 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50041E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50041E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.27528E+03 ;
RUNNING_TIME              (idx, 1)        =  1.29122E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.73417E-01  1.31633E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.27384E+02  3.62346E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.61333E-02  1.86000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.53217E-01  4.96667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.29116E+02  5.23427E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87657 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98379E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88395E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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
URES_USED                 (idx, 1)        = 174 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.49837E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66943E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.14753E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89057E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32051E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57649E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50601E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.80185E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.55850E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.00542E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.15259E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79534E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74322E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48841E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36279E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34616E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56019E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39254E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42584E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07753E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54041E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.14371E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.65926E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.75741E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12670E-03 -2.48773E+26  2.21046E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.10776E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.58550E+18 0.00401  1.69159E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  9.18763E+19 0.00054  9.80315E-01 7.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.25074E+16 0.02259  4.53480E-04 0.02256 ];
TH232_CAPT                (idx, [1:   4]) = [  9.52918E+19 0.00053  7.42354E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34087E+19 0.00142  1.04459E-01 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.45613E+16 0.04117  1.13460E-04 0.04112 ];
SM149_CAPT                (idx, [1:   4]) = [  9.74228E+16 0.01496  7.58844E-04 0.01493 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001633 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16992E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001633 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5256797 5.26226E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3838336 3.84209E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906500 9.07354E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001633 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.45286E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36412E+20 3.8E-06  2.36412E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37750E+19 1.2E-07  9.37750E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28373E+20 0.00031  1.14652E+20 0.00023  1.37208E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22148E+20 0.00018  2.08427E+20 0.00012  1.37208E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43935E+20 0.00037  2.43935E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.23989E+22 0.00026  3.07391E+22 0.00024  1.65978E+21 0.00169 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.21346E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44283E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09576E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51441E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51441E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.60719E-01 0.04674 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33122E-01 0.01117 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.22477E-04 0.02142 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.28529E+03 0.02358 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09269E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.16106E-01 0.03494 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.51099E-01 0.03494 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52106E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.68470E-01 0.00045  6.03429E-02 0.00044  1.95139E-04 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.68911E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.69213E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.68911E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06557E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06112E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06154E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28719E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28644E-02 0.00101 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49049E-01 0.00163 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48696E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33865E-03 0.00543  2.69598E-04 0.01958  8.24116E-04 0.01062  6.15248E-04 0.01321  1.31496E-03 0.00941  2.65902E-04 0.01921  4.88251E-05 0.04665 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.39014E-01 0.01486  1.23817E-02 0.00435  3.23314E-02 9.1E-05  1.06108E-01 0.00058  2.97375E-01 0.00026  1.23259E+00 0.00252  4.49999E+00 0.04371 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23981E-03 0.00875  2.60077E-04 0.02922  8.04343E-04 0.01609  6.07142E-04 0.01943  1.26488E-03 0.01405  2.57399E-04 0.02854  4.59647E-05 0.07475 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.36316E-01 0.02399  1.24753E-02 4.0E-05  3.23257E-02 0.00012  1.06111E-01 0.00082  2.97379E-01 0.00039  1.23531E+00 0.00034  6.68262E+00 0.02840 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.70178E-07 0.00126  8.69892E-07 0.00126  9.58128E-07 0.01637 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.42673E-07 0.00117  8.42395E-07 0.00117  9.27850E-07 0.01638 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22461E-03 0.00891  2.52718E-04 0.03403  8.09983E-04 0.01787  6.02051E-04 0.02107  1.26196E-03 0.01509  2.54957E-04 0.03212  4.29398E-05 0.07425 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.34154E-01 0.02447  1.24756E-02 5.2E-05  3.23261E-02 0.00015  1.06264E-01 0.00109  2.97433E-01 0.00046  1.23546E+00 0.00040  6.71172E+00 0.04255 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.70320E-07 0.01729  7.70304E-07 0.01729  7.55689E-07 0.05041 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.46215E-07 0.01728  7.46198E-07 0.01728  7.32427E-07 0.05044 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87438E-03 0.04149  2.23489E-04 0.13774  7.73522E-04 0.07603  5.35593E-04 0.10290  1.06829E-03 0.06445  2.36191E-04 0.12563  3.72960E-05 0.29102 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.32742E-01 0.08622  1.24764E-02 0.00013  3.23283E-02 0.00046  1.05980E-01 0.00305  2.96806E-01 0.00155  1.23481E+00 0.00127  5.60108E+00 0.14153 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85707E-03 0.04068  2.14132E-04 0.13854  7.72867E-04 0.07455  5.35707E-04 0.10303  1.05929E-03 0.06282  2.38187E-04 0.12215  3.68847E-05 0.29384 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.33867E-01 0.08644  1.24766E-02 0.00012  3.23304E-02 0.00047  1.05990E-01 0.00306  2.96840E-01 0.00155  1.23481E+00 0.00127  5.60108E+00 0.14153 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.74278E+03 0.03802 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.63876E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.36577E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22102E-03 0.00587 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.72985E+03 0.00595 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94741E-09 0.00087 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92555E-05 0.00720  6.92743E-05 0.00724  9.47479E-07 0.45712 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.34353E-05 0.02417  6.32992E-05 0.02434  9.10843E-07 0.53603 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.16008E-04 0.02077  2.16227E-04 0.02090  1.55090E-04 0.44557 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74624E+01 0.01199 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48708E+01 0.00027  4.07662E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 18:14:27 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00723E+00  9.93900E-01  9.98952E-01  9.96449E-01  1.00205E+00  9.96887E-01  9.99083E-01  9.98286E-01  1.00519E+00  9.98318E-01  1.00285E+00  1.00732E+00  9.97403E-01  1.00593E+00  9.95730E-01  9.94796E-01  9.99397E-01  1.00023E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.63890E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63611E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39547E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53280E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63022E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51416E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50510E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24716E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.15069E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500315 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50047E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50047E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.64964E+03 ;
RUNNING_TIME              (idx, 1)        =  1.66744E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.17250E-01  1.43833E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.64829E+02  3.74451E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.45000E-02  1.83667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.08567E-01  5.53000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.66738E+02  5.30706E+02 ];
CPU_USAGE                 (idx, 1)        = 9.89321 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98406E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90142E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.52585E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66423E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.52624E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91300E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33455E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58132E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49921E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.98327E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.67948E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.27761E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.25366E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69997E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85400E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93713E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32739E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31897E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52280E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34153E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81498E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07546E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51327E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50429E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70662E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.88957E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.28596E-03 -5.04734E+26  2.21302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.34714E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.59931E+18 0.00387  1.70728E-02 0.00381 ];
U233_FISS                 (idx, [1:   4]) = [  9.15003E+19 0.00050  9.76811E-01 8.1E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.61273E+17 0.01297  1.72166E-03 0.01296 ];
TH232_CAPT                (idx, [1:   4]) = [  9.69585E+19 0.00054  7.37778E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33850E+19 0.00129  1.01853E-01 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  5.51941E+16 0.02120  4.19965E-04 0.02119 ];
SM149_CAPT                (idx, [1:   4]) = [  1.40240E+17 0.01389  1.06713E-03 0.01388 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001878 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15784E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001878 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5310140 5.31549E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3785133 3.78877E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906605 9.07316E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001878 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.42727E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36401E+20 3.8E-06  2.36401E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.3E-07  9.37729E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31383E+20 0.00029  1.17392E+20 0.00021  1.39915E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.25156E+20 0.00017  2.11165E+20 0.00012  1.39915E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47239E+20 0.00034  2.47239E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29176E+22 0.00026  3.12261E+22 0.00022  1.69146E+21 0.00199 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24327E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47589E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11507E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51186E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51186E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.50653E-01 0.05075 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17935E-01 0.01344 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.97020E-04 0.02511 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.46945E+03 0.02271 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09272E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.51351E-01 0.03920 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.92158E-01 0.03920 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52099E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.55174E-01 0.00043  5.95054E-02 0.00042  1.90730E-04 0.00876 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55923E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.56206E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55923E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05128E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06156E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06189E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28659E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28595E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51268E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51152E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39757E-03 0.00609  2.65651E-04 0.02017  8.36666E-04 0.01213  6.18254E-04 0.01249  1.34047E-03 0.00916  2.85813E-04 0.01855  5.07165E-05 0.04664 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47111E-01 0.01581  1.23818E-02 0.00435  3.23278E-02 9.4E-05  1.06118E-01 0.00060  2.97571E-01 0.00027  1.23271E+00 0.00252  4.49245E+00 0.04332 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22999E-03 0.00830  2.51039E-04 0.02980  7.87920E-04 0.01697  5.93622E-04 0.01750  1.28319E-03 0.01331  2.66178E-04 0.02576  4.80379E-05 0.06211 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.43266E-01 0.02134  1.24756E-02 4.1E-05  3.23274E-02 0.00012  1.06099E-01 0.00087  2.97475E-01 0.00037  1.23589E+00 0.00036  6.38740E+00 0.02959 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.83640E-07 0.00120  8.83278E-07 0.00120  9.98626E-07 0.01707 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.43969E-07 0.00112  8.43623E-07 0.00112  9.53547E-07 0.01699 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19495E-03 0.00887  2.53638E-04 0.03005  7.78757E-04 0.01909  5.78003E-04 0.02042  1.26669E-03 0.01395  2.66291E-04 0.03120  5.15728E-05 0.07078 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.55983E-01 0.02517  1.24754E-02 4.9E-05  3.23207E-02 0.00015  1.06266E-01 0.00110  2.97649E-01 0.00049  1.23678E+00 0.00053  6.48309E+00 0.04025 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.82649E-07 0.01708  7.82361E-07 0.01708  8.52586E-07 0.05725 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.47617E-07 0.01708  7.47340E-07 0.01708  8.14231E-07 0.05715 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.65365E-03 0.04588  2.05138E-04 0.15175  6.76123E-04 0.07877  4.53201E-04 0.09846  1.05420E-03 0.07016  2.23587E-04 0.14035  4.14030E-05 0.23360 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.51838E-01 0.10467  1.24757E-02 0.00014  3.23164E-02 0.00046  1.06487E-01 0.00371  2.97186E-01 0.00154  1.23572E+00 0.00162  5.95597E+00 0.11326 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.64013E-03 0.04522  2.04290E-04 0.14907  6.64227E-04 0.07923  4.52112E-04 0.09760  1.06044E-03 0.06968  2.14835E-04 0.13782  4.42283E-05 0.23084 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.58003E-01 0.10360  1.24757E-02 0.00013  3.23187E-02 0.00047  1.06501E-01 0.00373  2.97191E-01 0.00153  1.23568E+00 0.00162  5.92463E+00 0.11346 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.40274E+03 0.04268 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.77082E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.37702E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.11628E-03 0.00619 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.55388E+03 0.00625 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93223E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74570E-05 0.00861  6.74425E-05 0.00860  6.75378E-07 0.52071 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.19487E-05 0.02661  6.19876E-05 0.02662  5.66486E-07 0.56276 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.91256E-04 0.02424  1.91508E-04 0.02427  1.35874E-04 0.50710 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69774E+01 0.01187 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50510E+01 0.00025  4.08684E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 18:52:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01097E+00  9.97113E-01  9.98557E-01  1.00613E+00  9.93109E-01  9.96329E-01  9.96917E-01  1.00155E+00  1.00185E+00  1.00562E+00  9.96048E-01  9.98171E-01  9.90562E-01  9.99563E-01  9.97564E-01  1.00451E+00  1.00022E+00  1.00522E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.63316E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63668E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39524E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53237E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63106E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52820E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51916E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27446E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.15364E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500216 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50043E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50043E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.02745E+03 ;
RUNNING_TIME              (idx, 1)        =  2.04711E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.60300E-01  1.43050E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.02617E+02  3.77884E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.23333E-02  1.78333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.60233E-01  5.16333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.04705E+02  5.43159E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90397 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98413E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91275E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.59424E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67394E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.30995E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94816E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35778E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61410E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50615E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09563E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.78998E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.68556E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.41489E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25597E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94820E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35264E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33770E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32656E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53309E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.01204E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17366E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11507E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54633E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.32584E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.80813E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.99857E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.42811E-03 -7.56917E+26  2.21554E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.52986E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59997E+18 0.00388  1.70667E-02 0.00382 ];
U233_FISS                 (idx, [1:   4]) = [  9.12030E+19 0.00056  9.72872E-01 8.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.39169E+17 0.00846  3.61878E-03 0.00852 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81990E+19 0.00052  7.33453E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33609E+19 0.00136  9.97943E-02 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  1.24148E+17 0.01403  9.27443E-04 0.01406 ];
SM149_CAPT                (idx, [1:   4]) = [  1.56959E+17 0.01198  1.17249E-03 0.01200 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001731 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14778E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001731 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5350836 5.35625E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3746882 3.75039E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904013 9.04835E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001731 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -8.00937E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36386E+20 3.6E-06  2.36386E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37700E+19 1.1E-07  9.37700E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33860E+20 0.00028  1.19577E+20 0.00021  1.42832E+19 0.00185 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27630E+20 0.00017  2.13347E+20 0.00012  1.42832E+19 0.00185 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49964E+20 0.00035  2.49964E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.33505E+22 0.00025  3.16276E+22 0.00023  1.72296E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.26183E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.50248E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.13085E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50804E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50804E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.36298E-01 0.05322 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.28939E-01 0.01264 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90345E-04 0.02421 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.30057E+03 0.02382 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09520E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.13543E-01 0.04174 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.57933E-01 0.04174 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52091E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.45370E-01 0.00045  5.89010E-02 0.00044  1.87705E-04 0.00893 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.45711E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.45725E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.45711E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03980E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06110E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05992E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28718E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28851E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52562E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53430E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.43856E-03 0.00582  2.80836E-04 0.01817  8.49145E-04 0.01164  6.21949E-04 0.01293  1.34025E-03 0.00903  2.87602E-04 0.01922  5.87789E-05 0.04130 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.57485E-01 0.01452  1.24753E-02 3.2E-05  3.23242E-02 9.6E-05  1.06193E-01 0.00060  2.97598E-01 0.00027  1.23770E+00 0.00030  4.87592E+00 0.03799 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26200E-03 0.00836  2.59323E-04 0.02754  7.98989E-04 0.01642  5.85788E-04 0.01920  1.28378E-03 0.01331  2.77355E-04 0.02927  5.67602E-05 0.06020 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.65233E-01 0.02257  1.24757E-02 3.4E-05  3.23266E-02 0.00013  1.06324E-01 0.00093  2.97719E-01 0.00042  1.23814E+00 0.00041  6.50755E+00 0.02691 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.93442E-07 0.00123  8.93036E-07 0.00123  1.02109E-06 0.01670 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.44556E-07 0.00112  8.44172E-07 0.00113  9.65189E-07 0.01670 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18490E-03 0.00903  2.51993E-04 0.03110  7.95168E-04 0.01884  5.77325E-04 0.02199  1.23383E-03 0.01537  2.66490E-04 0.03181  6.00996E-05 0.06538 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.69932E-01 0.02468  1.24753E-02 5.6E-05  3.23209E-02 0.00016  1.06040E-01 0.00098  2.97623E-01 0.00048  1.23702E+00 0.00056  6.65215E+00 0.03641 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.91138E-07 0.01723  7.90824E-07 0.01723  8.34342E-07 0.05663 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.47813E-07 0.01722  7.47519E-07 0.01723  7.88330E-07 0.05658 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.74009E-03 0.04333  2.30518E-04 0.14517  6.43417E-04 0.08711  4.77432E-04 0.09106  1.05453E-03 0.06240  2.75721E-04 0.12604  5.84724E-05 0.24396 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.32468E-01 0.10718  1.24730E-02 0.00019  3.23206E-02 0.00061  1.05712E-01 0.00283  2.98031E-01 0.00165  1.23735E+00 0.00156  6.59793E+00 0.09395 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.71885E-03 0.04235  2.22111E-04 0.14050  6.25073E-04 0.08457  4.76519E-04 0.09075  1.06017E-03 0.06163  2.75447E-04 0.12414  5.95318E-05 0.23592 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.33546E-01 0.10427  1.24731E-02 0.00019  3.23199E-02 0.00062  1.05761E-01 0.00289  2.98046E-01 0.00164  1.23730E+00 0.00156  6.59793E+00 0.09395 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.48947E+03 0.04032 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.87644E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.39085E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.15037E-03 0.00546 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.54949E+03 0.00544 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93366E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71293E-05 0.00741  6.71395E-05 0.00742  8.01153E-07 0.44738 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.97827E-05 0.02802  5.96614E-05 0.02806  1.31798E-06 0.49594 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84716E-04 0.02333  1.84848E-04 0.02330  1.61910E-04 0.45074 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70698E+01 0.01116 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51916E+01 0.00026  4.09059E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 19:30:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00075E+00  1.00250E+00  9.96121E-01  9.99361E-01  9.89583E-01  1.00171E+00  9.99825E-01  9.98780E-01  1.00370E+00  1.00368E+00  1.00487E+00  1.00615E+00  9.91242E-01  9.97630E-01  9.99132E-01  1.00357E+00  9.93234E-01  1.00816E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.63167E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63683E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39069E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52785E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62900E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53786E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52878E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.30846E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16032E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500433 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50058E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50058E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.40968E+03 ;
RUNNING_TIME              (idx, 1)        =  2.43147E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.00667E+00  1.46367E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.40860E+02  3.82428E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.11000E-01  1.86667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.11383E-01  5.11000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.43141E+02  5.46768E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91042 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98276E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92024E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.62216E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67204E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.34435E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97530E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37396E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61446E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50224E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26717E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.92379E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.68629E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.56737E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93990E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06615E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07114E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.31033E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30462E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50345E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.82751E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78735E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.12019E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55520E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.89352E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85374E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.01495E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43284E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.80151E-03 -1.28096E+27  2.22078E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.81797E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.62476E+18 0.00395  1.73167E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  9.04013E+19 0.00052  9.63465E-01 1.0E-04 ];
U235_FISS                 (idx, [1:   4]) = [  8.35588E+17 0.00567  8.90495E-03 0.00562 ];
TH232_CAPT                (idx, [1:   4]) = [  9.98382E+19 0.00050  7.28137E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32174E+19 0.00136  9.63972E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  2.98378E+17 0.00940  2.17637E-03 0.00941 ];
SM149_CAPT                (idx, [1:   4]) = [  1.70667E+17 0.01250  1.24457E-03 0.01247 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002304 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13823E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002304 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5398754 5.40381E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3694662 3.69790E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908888 9.09665E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002304 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.77420E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36350E+20 3.5E-06  2.36350E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37626E+19 1.1E-07  9.37626E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37145E+20 0.00031  1.22654E+20 0.00020  1.44911E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30908E+20 0.00018  2.16417E+20 0.00012  1.44911E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53739E+20 0.00033  2.53739E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.39314E+22 0.00025  3.21870E+22 0.00022  1.74430E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.30827E+19 0.00162 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53990E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15036E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50046E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50046E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.62336E-01 0.05518 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21298E-01 0.01436 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.86308E-04 0.02492 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.25146E+03 0.02394 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09036E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.94599E-01 0.04260 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.40614E-01 0.04260 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52073E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99702E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.32040E-01 0.00044  5.80661E-02 0.00044  1.91929E-04 0.00884 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.31632E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.31513E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.31632E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02488E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05347E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05440E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29704E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29561E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.57580E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.57859E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.56501E-03 0.00519  2.77037E-04 0.01963  8.77085E-04 0.01079  6.59879E-04 0.01223  1.39096E-03 0.00855  2.99506E-04 0.01896  6.05371E-05 0.04250 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.59830E-01 0.01509  1.24441E-02 0.00251  3.23176E-02 0.00010  1.06108E-01 0.00055  2.98005E-01 0.00027  1.23887E+00 0.00042  5.02676E+00 0.03782 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27721E-03 0.00788  2.57376E-04 0.02733  8.06770E-04 0.01618  6.08944E-04 0.01929  1.27366E-03 0.01288  2.72837E-04 0.02581  5.76237E-05 0.06393 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.67654E-01 0.02420  1.24756E-02 4.0E-05  3.23116E-02 0.00014  1.06129E-01 0.00076  2.98157E-01 0.00041  1.23878E+00 0.00054  6.73043E+00 0.02616 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.07851E-07 0.00121  9.07564E-07 0.00122  9.85864E-07 0.01740 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.46083E-07 0.00112  8.45815E-07 0.00112  9.18987E-07 0.01743 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.29542E-03 0.00880  2.54330E-04 0.03340  8.27237E-04 0.01834  6.04914E-04 0.02168  1.27666E-03 0.01478  2.77640E-04 0.03008  5.46349E-05 0.06833 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.57947E-01 0.02448  1.24751E-02 6.1E-05  3.23202E-02 0.00018  1.06048E-01 0.00096  2.97838E-01 0.00047  1.23912E+00 0.00074  6.80665E+00 0.03623 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.05334E-07 0.01710  8.05195E-07 0.01710  8.11347E-07 0.05093 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.50606E-07 0.01710  7.50472E-07 0.01710  7.56728E-07 0.05099 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.79143E-03 0.04335  2.26634E-04 0.13550  7.23362E-04 0.08441  4.87148E-04 0.08862  1.08265E-03 0.07018  2.12056E-04 0.13733  5.95776E-05 0.32009 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.98583E-01 0.11782  1.24776E-02 0.00010  3.23406E-02 0.00065  1.05897E-01 0.00295  2.96998E-01 0.00148  1.23717E+00 0.00163  7.43404E+00 0.09116 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.79299E-03 0.04291  2.28504E-04 0.13293  7.23335E-04 0.08332  4.96196E-04 0.08767  1.07632E-03 0.06969  2.08727E-04 0.13145  5.99093E-05 0.30906 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.98628E-01 0.11453  1.24776E-02 0.00010  3.23401E-02 0.00065  1.05890E-01 0.00293  2.97027E-01 0.00149  1.23718E+00 0.00163  7.43404E+00 0.09116 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.51008E+03 0.04068 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.01239E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.39917E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28707E-03 0.00620 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.64821E+03 0.00627 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94389E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73506E-05 0.00875  6.71663E-05 0.00911  1.43386E-06 0.35488 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.69460E-05 0.02632  5.68545E-05 0.02648  1.01491E-06 0.49736 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79559E-04 0.02419  1.79396E-04 0.02425  2.25653E-04 0.35148 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69000E+01 0.01129 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52878E+01 0.00027  4.09010E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 20:09:47 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00648E+00  9.94756E-01  9.91972E-01  9.98983E-01  9.95664E-01  1.01037E+00  9.97977E-01  1.00300E+00  1.00788E+00  9.98382E-01  9.98493E-01  1.00041E+00  1.00431E+00  9.99395E-01  9.95971E-01  1.00423E+00  9.96722E-01  9.94997E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.63729E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63627E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38661E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52402E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63338E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54367E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53456E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33308E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16959E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500587 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50071E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50071E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.79113E+03 ;
RUNNING_TIME              (idx, 1)        =  2.82082E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.15498E+00  1.48317E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.79606E+02  3.87454E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.30950E-01  1.99500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.12470E+00  7.13267E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.81415E+02  5.50292E+02 ];
CPU_USAGE                 (idx, 1)        = 9.89474 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98398E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90256E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.70164E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68524E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.05226E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00508E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38913E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.66350E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51332E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75110E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22490E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.11673E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.82273E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23238E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33943E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32969E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30663E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29841E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49676E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.54817E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27982E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.15791E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.66053E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84035E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94122E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03172E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.18153E-02 -2.60878E+27  2.23406E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01684E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.62142E+18 0.00392  1.73004E-02 0.00386 ];
U233_FISS                 (idx, [1:   4]) = [  8.80512E+19 0.00051  9.39540E-01 0.00012 ];
U235_FISS                 (idx, [1:   4]) = [  2.34047E+18 0.00321  2.49740E-02 0.00318 ];
PU239_FISS                (idx, [1:   4]) = [  7.73305E+14 0.18212  8.27139E-06 0.18215 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00770E+20 0.00053  7.14712E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.29206E+19 0.00143  9.16404E-02 0.00139 ];
U235_CAPT                 (idx, [1:   4]) = [  8.26139E+17 0.00544  5.85921E-03 0.00540 ];
PU239_CAPT                (idx, [1:   4]) = [  4.63209E+14 0.23065  3.29416E-06 0.23065 ];
PU240_CAPT                (idx, [1:   4]) = [  5.07304E+13 0.70625  3.58296E-07 0.70629 ];
SM149_CAPT                (idx, [1:   4]) = [  1.73409E+17 0.01298  1.23026E-03 0.01302 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002852 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15025E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002852 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5461425 5.46638E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3630508 3.63347E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910919 9.11661E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002852 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 6.01634E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36254E+20 4.0E-06  2.36254E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37405E+19 1.1E-07  9.37405E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.41053E+20 0.00028  1.26315E+20 0.00021  1.47384E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34794E+20 0.00017  2.20055E+20 0.00012  1.47384E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57931E+20 0.00037  2.57931E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45541E+22 0.00025  3.27836E+22 0.00023  1.77044E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35160E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58310E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17087E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.48114E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.48114E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.40414E-01 0.05444 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.25841E-01 0.01352 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.81160E-04 0.02516 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.45204E+03 0.02172 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08836E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.71577E-01 0.04371 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.19438E-01 0.04371 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52029E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99749E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.15702E-01 0.00042  5.70453E-02 0.00042  1.89455E-04 0.00873 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.15670E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.16005E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.15670E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00749E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02695E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02593E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33200E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33308E-02 0.00100 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.67347E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.67570E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.69551E-03 0.00565  2.86863E-04 0.01913  8.92804E-04 0.01137  6.83636E-04 0.01304  1.45965E-03 0.00859  3.08641E-04 0.01822  6.39137E-05 0.04082 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.61315E-01 0.01505  1.24749E-02 3.6E-05  3.22958E-02 0.00011  1.06290E-01 0.00056  2.98612E-01 0.00029  1.24568E+00 0.00058  4.84750E+00 0.03620 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.33191E-03 0.00807  2.58745E-04 0.02853  8.16316E-04 0.01586  6.12828E-04 0.01861  1.30763E-03 0.01252  2.77760E-04 0.02908  5.86336E-05 0.05807 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.66090E-01 0.02131  1.24752E-02 4.1E-05  3.22938E-02 0.00016  1.06286E-01 0.00080  2.98530E-01 0.00042  1.24615E+00 0.00080  6.32554E+00 0.02563 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.16956E-07 0.00128  9.16475E-07 0.00128  1.05702E-06 0.02748 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39605E-07 0.00122  8.39165E-07 0.00122  9.67910E-07 0.02748 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.31338E-03 0.00895  2.53155E-04 0.03273  8.16571E-04 0.01898  6.06994E-04 0.02145  1.30533E-03 0.01403  2.71925E-04 0.03264  5.94090E-05 0.06717 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.64455E-01 0.02659  1.24754E-02 5.8E-05  3.22914E-02 0.00020  1.06103E-01 0.00089  2.98603E-01 0.00048  1.24695E+00 0.00111  6.30513E+00 0.03810 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.04881E-07 0.01706  8.04476E-07 0.01706  8.65718E-07 0.05376 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.37156E-07 0.01706  7.36786E-07 0.01706  7.92493E-07 0.05369 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.69455E-03 0.04465  2.13504E-04 0.15836  6.88308E-04 0.07784  4.98208E-04 0.09043  1.06388E-03 0.06637  1.83822E-04 0.13454  4.68302E-05 0.33299 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.43951E-01 0.08786  1.24788E-02 4.5E-05  3.22946E-02 0.00066  1.06090E-01 0.00293  2.99470E-01 0.00188  1.24349E+00 0.00315  8.01887E+00 0.09012 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.68032E-03 0.04340  2.06817E-04 0.15474  6.81781E-04 0.07630  4.93674E-04 0.08972  1.07011E-03 0.06472  1.80772E-04 0.13105  4.71713E-05 0.32392 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.46171E-01 0.08707  1.24788E-02 4.4E-05  3.22958E-02 0.00067  1.06102E-01 0.00294  2.99478E-01 0.00187  1.24342E+00 0.00313  7.99813E+00 0.09101 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.36476E+03 0.04178 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.08772E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.32103E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24058E-03 0.00501 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.56634E+03 0.00503 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95899E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68013E-05 0.00912  6.68196E-05 0.00913  1.81013E-06 0.33371 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.89833E-05 0.02820  5.89539E-05 0.02834  1.20240E-06 0.37069 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.74330E-04 0.02460  1.73971E-04 0.02468  2.53775E-04 0.31395 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68644E+01 0.01126 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53456E+01 0.00028  4.07230E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 20:49:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01007E+00  9.93032E-01  9.92823E-01  1.00667E+00  9.97958E-01  1.00490E+00  9.99245E-01  1.00654E+00  9.99062E-01  9.89635E-01  1.00001E+00  1.00262E+00  9.91947E-01  1.00237E+00  9.91314E-01  1.00460E+00  9.99997E-01  1.00720E+00  ];
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

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.64931E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63507E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38617E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52397E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62772E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52349E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51437E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.29619E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16923E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500447 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50033E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50033E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.17683E+03 ;
RUNNING_TIME              (idx, 1)        =  3.21408E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.30432E+00  1.49333E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.18737E+02  3.91314E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.49250E-01  1.83000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.79293E+00  6.68183E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.20784E+02  5.54262E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88412 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98392E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89057E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.87287E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73050E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59776E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01818E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39247E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82079E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55750E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.13370E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77467E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.05279E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20321E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04589E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84745E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38242E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37964E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34840E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56849E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.83003E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30602E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26204E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91696E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.42098E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09965E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03833E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.23791E-02 -4.94125E+27  2.25738E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02882E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.61285E+18 0.00406  1.72167E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  8.46180E+19 0.00054  9.03295E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.71726E+18 0.00235  5.03584E-02 0.00233 ];
U238_FISS                 (idx, [1:   4]) = [  2.55279E+13 1.00000  2.77377E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  2.18013E+16 0.03572  2.32783E-04 0.03573 ];
PU240_FISS                (idx, [1:   4]) = [  5.44729E+14 0.22275  5.80284E-06 0.22262 ];
PU241_FISS                (idx, [1:   4]) = [  7.04157E+14 0.20018  7.50035E-06 0.20010 ];
TH232_CAPT                (idx, [1:   4]) = [  9.87130E+19 0.00050  6.92732E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23508E+19 0.00134  8.66759E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  1.66070E+18 0.00382  1.16542E-02 0.00381 ];
U238_CAPT                 (idx, [1:   4]) = [  1.29873E+14 0.44500  9.10302E-07 0.44498 ];
PU239_CAPT                (idx, [1:   4]) = [  1.00978E+16 0.05105  7.08528E-05 0.05107 ];
PU240_CAPT                (idx, [1:   4]) = [  1.60791E+15 0.12763  1.13090E-05 0.12765 ];
PU241_CAPT                (idx, [1:   4]) = [  2.07667E+14 0.35047  1.45492E-06 0.35046 ];
SM149_CAPT                (idx, [1:   4]) = [  1.76985E+17 0.01145  1.24226E-03 0.01149 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001301 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13130E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001301 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5483837 5.48957E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605353 3.60879E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912111 9.12946E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001301 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.45058E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36128E+20 4.3E-06  2.36128E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37022E+19 1.1E-07  9.37022E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42566E+20 0.00029  1.27807E+20 0.00021  1.47594E+19 0.00185 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36269E+20 0.00018  2.21509E+20 0.00012  1.47594E+19 0.00185 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59582E+20 0.00034  2.59582E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46728E+22 0.00024  3.28991E+22 0.00021  1.77362E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36998E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59968E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17309E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42794E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42794E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.23009E-01 0.05482 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.29737E-01 0.01283 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.88399E-04 0.02454 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.86308E+03 0.02474 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08708E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.63023E-01 0.04415 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.11612E-01 0.04415 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51999E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99830E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09320E-01 0.00044  5.66432E-02 0.00043  1.94398E-04 0.00910 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09339E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09690E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09339E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00066E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.96631E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.96819E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.41530E-02 0.00148 ];
IMP_EALF                  (idx, [1:   2]) = [  1.41227E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81758E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81322E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.82048E-03 0.00548  2.83308E-04 0.02001  9.10511E-04 0.01149  6.87616E-04 0.01208  1.53460E-03 0.00838  3.27188E-04 0.01818  7.72622E-05 0.03769 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.93509E-01 0.01502  1.24761E-02 2.8E-05  3.22724E-02 0.00011  1.06544E-01 0.00062  3.00062E-01 0.00032  1.25056E+00 0.00261  5.54965E+00 0.03112 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42208E-03 0.00771  2.55058E-04 0.02655  8.18125E-04 0.01689  6.17768E-04 0.01792  1.36496E-03 0.01200  2.97462E-04 0.02679  6.87009E-05 0.05650 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.93467E-01 0.02127  1.24761E-02 3.7E-05  3.22766E-02 0.00017  1.06573E-01 0.00083  3.00128E-01 0.00045  1.25303E+00 0.00094  6.83189E+00 0.02179 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.98369E-07 0.00124  8.97934E-07 0.00125  1.03111E-06 0.01802 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16844E-07 0.00117  8.16448E-07 0.00117  9.37517E-07 0.01801 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41915E-03 0.00912  2.49190E-04 0.03344  8.23916E-04 0.01845  6.18495E-04 0.02237  1.35392E-03 0.01470  3.00141E-04 0.03098  7.34918E-05 0.06267 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.97480E-01 0.02549  1.24761E-02 4.7E-05  3.22714E-02 0.00020  1.06567E-01 0.00103  2.99829E-01 0.00050  1.25501E+00 0.00127  6.54676E+00 0.03177 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.91858E-07 0.01713  7.91421E-07 0.01713  8.97921E-07 0.05805 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.19929E-07 0.01712  7.19534E-07 0.01712  8.15909E-07 0.05809 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.07697E-03 0.04175  2.62703E-04 0.13977  6.55636E-04 0.08878  5.91827E-04 0.08657  1.25647E-03 0.06023  2.37395E-04 0.12961  7.29421E-05 0.22427 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.23380E-01 0.11194  1.24769E-02 0.00012  3.22744E-02 0.00074  1.06574E-01 0.00311  2.99231E-01 0.00173  1.25357E+00 0.00358  6.84813E+00 0.08586 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08198E-03 0.04206  2.58002E-04 0.14415  6.51279E-04 0.08883  5.90790E-04 0.08637  1.25577E-03 0.06136  2.46848E-04 0.12611  7.92952E-05 0.22403 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.26798E-01 0.10745  1.24770E-02 0.00012  3.22729E-02 0.00073  1.06582E-01 0.00310  2.99258E-01 0.00173  1.25353E+00 0.00358  6.84813E+00 0.08586 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.90062E+03 0.03839 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.90645E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09809E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42062E-03 0.00631 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84183E+03 0.00640 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92738E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63798E-05 0.00913  6.64978E-05 0.00917  1.88787E-06 0.27815 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.90856E-05 0.02895  5.92117E-05 0.02915  1.58549E-06 0.33327 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79997E-04 0.02378  1.79383E-04 0.02392  3.30155E-04 0.27437 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63195E+01 0.01057 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51437E+01 0.00025  4.01724E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 21:28:45 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00820E+00  1.00265E+00  1.00392E+00  9.99795E-01  1.00123E+00  1.00028E+00  9.93339E-01  1.00136E+00  1.00207E+00  1.00443E+00  1.00587E+00  9.89086E-01  9.96554E-01  9.98077E-01  9.93758E-01  1.00466E+00  9.95424E-01  9.99299E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.65670E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63433E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38650E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52451E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62943E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51233E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50322E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27373E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16904E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499865 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50052E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50052E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.56623E+03 ;
RUNNING_TIME              (idx, 1)        =  3.61055E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.45845E+00  1.54133E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.58185E+02  3.94484E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.67767E-01  1.85167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.45000E-02  1.45000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.40545E+00  6.12467E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.60489E+02  5.57009E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87725 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98484E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88286E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.90340E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73868E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.71777E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01257E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38801E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85671E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56591E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.08689E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98985E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.91186E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.38969E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13512E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04301E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81895E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38624E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35204E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57421E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88132E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73615E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27651E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97520E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.86089E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11858E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03857E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.74773E-02 -6.06692E+27  2.26864E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02616E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59888E+18 0.00418  1.70646E-02 0.00414 ];
U233_FISS                 (idx, [1:   4]) = [  8.35530E+19 0.00059  8.91750E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.41602E+18 0.00221  5.78061E-02 0.00217 ];
U238_FISS                 (idx, [1:   4]) = [  5.11133E+13 0.70626  5.43714E-07 0.70632 ];
PU239_FISS                (idx, [1:   4]) = [  4.99443E+16 0.02190  5.32993E-04 0.02190 ];
PU240_FISS                (idx, [1:   4]) = [  1.79068E+15 0.11159  1.90510E-05 0.11157 ];
PU241_FISS                (idx, [1:   4]) = [  1.96926E+15 0.11616  2.09826E-05 0.11606 ];
TH232_CAPT                (idx, [1:   4]) = [  9.75403E+19 0.00053  6.84134E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22179E+19 0.00152  8.56947E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  1.92767E+18 0.00350  1.35209E-02 0.00350 ];
U238_CAPT                 (idx, [1:   4]) = [  2.32718E+14 0.33001  1.62531E-06 0.33000 ];
PU239_CAPT                (idx, [1:   4]) = [  2.41648E+16 0.03405  1.69526E-04 0.03402 ];
PU240_CAPT                (idx, [1:   4]) = [  5.37546E+15 0.06421  3.76912E-05 0.06418 ];
PU241_CAPT                (idx, [1:   4]) = [  3.09342E+14 0.28469  2.17294E-06 0.28468 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81479E+17 0.01155  1.27291E-03 0.01155 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002090 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10599E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002090 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486074 5.49121E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605559 3.60863E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910457 9.11220E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002090 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.71946E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36119E+20 4.5E-06  2.36119E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36879E+19 1.1E-07  9.36879E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42525E+20 0.00028  1.27774E+20 0.00020  1.47509E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36213E+20 0.00017  2.21462E+20 0.00012  1.47509E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59644E+20 0.00036  2.59644E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46233E+22 0.00025  3.28530E+22 0.00023  1.77027E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36600E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59873E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17046E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39994E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39994E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.41755E-01 0.05372 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21764E-01 0.01336 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.89181E-04 0.02296 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.79501E+03 0.02447 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08881E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.82840E-01 0.04239 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.29674E-01 0.04239 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52027E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99861E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09374E-01 0.00045  5.66463E-02 0.00045  1.95568E-04 0.00949 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09617E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09442E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09617E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00084E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94250E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94389E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44940E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44705E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86833E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86236E-01 0.00052 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85566E-03 0.00579  2.85505E-04 0.01983  9.02140E-04 0.01056  6.97979E-04 0.01204  1.53738E-03 0.00919  3.52891E-04 0.01748  7.97639E-05 0.03816 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05254E-01 0.01449  1.24449E-02 0.00251  3.22568E-02 0.00013  1.06592E-01 0.00053  3.00312E-01 0.00030  1.25644E+00 0.00074  5.77076E+00 0.02950 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43229E-03 0.00782  2.48178E-04 0.03016  8.08400E-04 0.01523  6.17099E-04 0.01737  1.37194E-03 0.01280  3.17338E-04 0.02525  6.93406E-05 0.05639 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09308E-01 0.02189  1.24758E-02 5.1E-05  3.22653E-02 0.00020  1.06546E-01 0.00074  3.00403E-01 0.00045  1.25630E+00 0.00095  7.02492E+00 0.02059 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.87777E-07 0.00134  8.87293E-07 0.00134  1.03192E-06 0.02138 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.07243E-07 0.00123  8.06803E-07 0.00123  9.38369E-07 0.02139 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43484E-03 0.00967  2.60122E-04 0.03441  8.00935E-04 0.01896  6.24830E-04 0.02035  1.36667E-03 0.01566  3.13096E-04 0.02882  6.91878E-05 0.06202 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.99768E-01 0.02383  1.24766E-02 6.9E-05  3.22611E-02 0.00023  1.06610E-01 0.00101  3.00318E-01 0.00052  1.25698E+00 0.00141  6.84078E+00 0.03006 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.83502E-07 0.01714  7.82913E-07 0.01715  8.98126E-07 0.04975 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.12539E-07 0.01713  7.12000E-07 0.01714  8.17615E-07 0.04986 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11825E-03 0.04429  2.29224E-04 0.12980  7.24047E-04 0.07697  5.44439E-04 0.09834  1.24243E-03 0.06152  3.10765E-04 0.12830  6.73473E-05 0.34226 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.85966E-01 0.10757  1.24772E-02 0.00013  3.22885E-02 0.00076  1.06671E-01 0.00329  3.00135E-01 0.00183  1.25828E+00 0.00355  5.85091E+00 0.11137 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.11440E-03 0.04341  2.25975E-04 0.12776  7.22307E-04 0.07556  5.50628E-04 0.09511  1.24278E-03 0.06049  3.08846E-04 0.12600  6.38632E-05 0.34247 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.81203E-01 0.10671  1.24771E-02 0.00013  3.22872E-02 0.00075  1.06667E-01 0.00327  3.00188E-01 0.00181  1.25859E+00 0.00356  5.85175E+00 0.11137 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.01750E+03 0.04160 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.80460E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.00603E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.48800E-03 0.00529 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.96230E+03 0.00534 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91649E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77775E-05 0.00788  6.77857E-05 0.00788  9.35698E-07 0.41882 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.14838E-05 0.02435  6.16062E-05 0.02435  5.44209E-07 0.51084 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.83765E-04 0.02257  1.83879E-04 0.02269  1.60814E-04 0.40702 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64127E+01 0.01151 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50322E+01 0.00026  3.99419E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 22:08:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00755E+00  1.00123E+00  1.00261E+00  1.00815E+00  9.96274E-01  9.98855E-01  9.96914E-01  1.00192E+00  1.00295E+00  1.00410E+00  1.00089E+00  9.93954E-01  9.98502E-01  9.97195E-01  9.89641E-01  1.00370E+00  9.95960E-01  9.99620E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.66173E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63383E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38613E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52429E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62844E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50369E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49457E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.25862E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16945E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499763 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50011E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50011E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.95902E+03 ;
RUNNING_TIME              (idx, 1)        =  4.01069E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.61218E+00  1.53733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.97997E+02  3.98116E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.85467E-01  1.77000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.81667E-02  1.36667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.04325E+00  6.37767E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.00477E+02  5.58956E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87117 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98471E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.87602E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.90289E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73929E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65884E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99730E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37737E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87144E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56757E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.17365E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17083E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.91859E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.55857E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21205E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20649E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20389E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37797E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34463E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56467E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90731E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11833E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26854E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99888E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.20661E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09731E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03868E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.26284E-02 -7.20426E+27  2.28002E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02350E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.57187E+18 0.00399  1.67725E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  8.27682E+19 0.00054  8.83193E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.92188E+18 0.00202  6.31904E-02 0.00196 ];
U238_FISS                 (idx, [1:   4]) = [  2.63205E+13 1.00000  2.80994E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  8.60035E+16 0.01749  9.17473E-04 0.01746 ];
PU240_FISS                (idx, [1:   4]) = [  3.86651E+15 0.08376  4.12948E-05 0.08379 ];
PU241_FISS                (idx, [1:   4]) = [  5.35172E+15 0.06770  5.70748E-05 0.06766 ];
TH232_CAPT                (idx, [1:   4]) = [  9.65090E+19 0.00053  6.77095E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20754E+19 0.00139  8.47198E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  2.10419E+18 0.00352  1.47626E-02 0.00348 ];
U238_CAPT                 (idx, [1:   4]) = [  3.89224E+14 0.25364  2.72385E-06 0.25364 ];
PU239_CAPT                (idx, [1:   4]) = [  4.17013E+16 0.02489  2.92658E-04 0.02491 ];
PU240_CAPT                (idx, [1:   4]) = [  1.01159E+16 0.04899  7.09939E-05 0.04897 ];
PU241_CAPT                (idx, [1:   4]) = [  1.11767E+15 0.15169  7.82671E-06 0.15177 ];
SM149_CAPT                (idx, [1:   4]) = [  1.79082E+17 0.01253  1.25643E-03 0.01253 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000430 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12154E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000430 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5483015 5.48905E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605234 3.60902E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912181 9.13143E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000430 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.71946E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36131E+20 4.5E-06  2.36131E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36753E+19 1.0E-07  9.36753E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42506E+20 0.00029  1.27781E+20 0.00020  1.47246E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36181E+20 0.00017  2.21457E+20 0.00012  1.47246E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59671E+20 0.00035  2.59671E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45901E+22 0.00025  3.28210E+22 0.00022  1.76915E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37130E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59894E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16835E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.37274E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.37274E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.76227E-01 0.05266 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.20015E-01 0.01363 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.86505E-04 0.02534 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.56866E+03 0.02671 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08688E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.75307E-01 0.04304 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.22769E-01 0.04304 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52074E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99888E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09531E-01 0.00043  5.66619E-02 0.00043  1.96598E-04 0.00889 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09590E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09391E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09590E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00102E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92369E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92455E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47680E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47528E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90109E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90155E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.89184E-03 0.00510  2.85602E-04 0.01835  9.14062E-04 0.01056  6.91247E-04 0.01289  1.56838E-03 0.00813  3.54095E-04 0.01798  7.84556E-05 0.03666 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.03831E-01 0.01407  1.24761E-02 3.1E-05  3.22433E-02 0.00012  1.06478E-01 0.00055  3.00593E-01 0.00029  1.25701E+00 0.00075  5.85294E+00 0.02925 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48447E-03 0.00749  2.48099E-04 0.02942  8.16585E-04 0.01605  6.18518E-04 0.01802  1.41849E-03 0.01206  3.13020E-04 0.02599  6.97616E-05 0.05554 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02872E-01 0.02185  1.24758E-02 4.1E-05  3.22429E-02 0.00017  1.06498E-01 0.00077  3.00778E-01 0.00044  1.25696E+00 0.00093  7.02514E+00 0.02055 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.79997E-07 0.00134  8.79486E-07 0.00134  1.02611E-06 0.01779 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.00329E-07 0.00127  7.99865E-07 0.00128  9.32896E-07 0.01772 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44291E-03 0.00896  2.43917E-04 0.03387  7.96297E-04 0.01921  6.07313E-04 0.02237  1.40775E-03 0.01440  3.14874E-04 0.02964  7.27555E-05 0.05797 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.11764E-01 0.02344  1.24760E-02 5.5E-05  3.22394E-02 0.00024  1.06712E-01 0.00118  3.00685E-01 0.00052  1.25865E+00 0.00130  6.94082E+00 0.02978 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.80365E-07 0.01723  7.80339E-07 0.01723  7.78259E-07 0.05383 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.09674E-07 0.01723  7.09648E-07 0.01723  7.08365E-07 0.05393 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.02361E-03 0.04471  2.64600E-04 0.12963  8.00292E-04 0.09603  4.98993E-04 0.09083  1.16392E-03 0.06325  2.37423E-04 0.14702  5.83790E-05 0.22880 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.30520E-01 0.12060  1.24710E-02 0.00022  3.22416E-02 0.00065  1.06636E-01 0.00342  3.00849E-01 0.00197  1.26141E+00 0.00364  6.71784E+00 0.08517 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00947E-03 0.04397  2.65387E-04 0.12721  8.01183E-04 0.09270  4.94501E-04 0.09033  1.16510E-03 0.06164  2.25261E-04 0.14316  5.80309E-05 0.22027 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.15375E-01 0.11431  1.24710E-02 0.00022  3.22411E-02 0.00065  1.06664E-01 0.00342  3.00907E-01 0.00197  1.26140E+00 0.00363  6.72062E+00 0.08466 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88580E+03 0.04137 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.73921E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.94800E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.45814E-03 0.00712 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95797E+03 0.00717 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90065E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77729E-05 0.00850  6.77930E-05 0.00850  7.87085E-07 0.44876 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.95009E-05 0.02568  5.95735E-05 0.02571  6.63489E-07 0.53049 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79472E-04 0.02458  1.79673E-04 0.02462  1.23794E-04 0.44571 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60628E+01 0.01144 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49457E+01 0.00027  3.97638E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 22:49:32 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01648E+00  1.00038E+00  9.96434E-01  1.00566E+00  9.99661E-01  1.00132E+00  1.00646E+00  1.00285E+00  9.99837E-01  1.00432E+00  9.90587E-01  1.00400E+00  1.00275E+00  9.92952E-01  9.87027E-01  9.97139E-01  9.93115E-01  9.99014E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.68237E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63176E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38240E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52117E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63504E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49167E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48257E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24783E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.18260E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2501079 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50084E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50084E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.35989E+03 ;
RUNNING_TIME              (idx, 1)        =  4.41833E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.77145E+00  1.59267E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.38550E+02  4.05526E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.03967E-01  1.85000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.81667E-02  1.36667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.60595E+00  5.62650E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.41315E+02  5.61283E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86773 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98470E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.87226E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.92565E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74560E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.99261E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96290E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35692E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92866E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57600E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.73606E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75213E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.24497E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03810E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44282E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70451E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34383E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37342E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33833E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55770E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95119E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27225E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26563E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05606E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41671E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07560E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04123E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.30802E-02 -1.17200E+28  2.32517E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02203E+00 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.53960E+18 0.00415  1.64384E-02 0.00408 ];
U233_FISS                 (idx, [1:   4]) = [  8.06897E+19 0.00060  8.61566E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.92384E+18 0.00186  7.39331E-02 0.00184 ];
U238_FISS                 (idx, [1:   4]) = [  1.80759E+14 0.37513  1.93687E-06 0.37515 ];
PU239_FISS                (idx, [1:   4]) = [  3.00922E+17 0.00934  3.21299E-03 0.00931 ];
PU240_FISS                (idx, [1:   4]) = [  2.36454E+16 0.03365  2.52415E-04 0.03363 ];
PU241_FISS                (idx, [1:   4]) = [  3.31745E+16 0.02862  3.54400E-04 0.02868 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44391E+19 0.00056  6.59465E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17513E+19 0.00144  8.20597E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  2.43526E+18 0.00337  1.70052E-02 0.00333 ];
U238_CAPT                 (idx, [1:   4]) = [  1.56501E+15 0.13467  1.09167E-05 0.13460 ];
PU239_CAPT                (idx, [1:   4]) = [  1.41369E+17 0.01378  9.87314E-04 0.01378 ];
PU240_CAPT                (idx, [1:   4]) = [  6.04196E+16 0.02066  4.21898E-04 0.02065 ];
PU241_CAPT                (idx, [1:   4]) = [  6.65930E+15 0.06504  4.65051E-05 0.06499 ];
SM149_CAPT                (idx, [1:   4]) = [  1.79649E+17 0.01137  1.25457E-03 0.01139 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003360 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11237E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003360 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5497080 5.50146E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3595165 3.59789E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911115 9.11781E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003360 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.03611E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36285E+20 4.8E-06  2.36285E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36357E+19 9.9E-08  9.36357E+19 9.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43237E+20 0.00029  1.28358E+20 0.00020  1.48793E+19 0.00185 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36873E+20 0.00017  2.21994E+20 0.00011  1.48793E+19 0.00185 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60307E+20 0.00037  2.60307E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46751E+22 0.00026  3.28910E+22 0.00022  1.78412E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37352E+19 0.00163 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60608E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16808E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26968E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26968E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.73644E-01 0.04696 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.40295E-01 0.01131 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.04685E-04 0.02327 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.22510E+03 0.02084 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08824E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.81993E-01 0.03415 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.19697E-01 0.03415 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52345E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99972E+02 9.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07756E-01 0.00048  5.65474E-02 0.00048  1.95811E-04 0.00885 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07680E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.07767E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07680E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98737E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88334E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88090E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53773E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.54114E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.99334E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.99811E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.92225E-03 0.00532  2.81785E-04 0.02102  9.28193E-04 0.01138  6.99922E-04 0.01236  1.57238E-03 0.00861  3.60830E-04 0.01682  7.91368E-05 0.04031 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.07175E-01 0.01494  1.24470E-02 0.00251  3.22200E-02 0.00015  1.06662E-01 0.00054  3.01232E-01 0.00032  1.25925E+00 0.00090  5.79536E+00 0.03049 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48119E-03 0.00773  2.55067E-04 0.02777  8.20227E-04 0.01613  6.06507E-04 0.01779  1.39742E-03 0.01296  3.30518E-04 0.02441  7.14555E-05 0.05547 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.13119E-01 0.02082  1.24783E-02 7.2E-05  3.22245E-02 0.00021  1.06640E-01 0.00075  3.01195E-01 0.00044  1.26058E+00 0.00120  7.19388E+00 0.02020 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64180E-07 0.00127  8.63749E-07 0.00127  9.90673E-07 0.01764 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.84394E-07 0.00118  7.84003E-07 0.00119  8.99257E-07 0.01761 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45119E-03 0.00890  2.41130E-04 0.03676  8.18431E-04 0.01947  6.03383E-04 0.02085  1.39101E-03 0.01448  3.28938E-04 0.02636  6.83002E-05 0.06655 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06989E-01 0.02509  1.24797E-02 0.00020  3.22136E-02 0.00026  1.06795E-01 0.00102  3.01237E-01 0.00054  1.26036E+00 0.00146  7.05308E+00 0.03071 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.65225E-07 0.01713  7.65062E-07 0.01713  7.81144E-07 0.05078 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.94688E-07 0.01712  6.94541E-07 0.01712  7.09158E-07 0.05076 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09545E-03 0.04320  1.90022E-04 0.16508  7.46181E-04 0.08720  4.99059E-04 0.09496  1.32879E-03 0.06595  2.93963E-04 0.12119  3.74314E-05 0.28256 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.87827E-01 0.10175  1.24760E-02 0.00017  3.21669E-02 0.00074  1.06785E-01 0.00316  3.00637E-01 0.00179  1.26054E+00 0.00359  6.80217E+00 0.10585 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09277E-03 0.04233  1.83594E-04 0.16032  7.43164E-04 0.08354  5.09020E-04 0.09372  1.33571E-03 0.06479  2.86975E-04 0.12010  3.43109E-05 0.27389 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.82713E-01 0.09800  1.24760E-02 0.00017  3.21674E-02 0.00073  1.06775E-01 0.00316  3.00555E-01 0.00176  1.26061E+00 0.00358  6.80217E+00 0.10585 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.05201E+03 0.03988 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.57209E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78062E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46040E-03 0.00654 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03657E+03 0.00648 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90942E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79198E-05 0.00710  6.79565E-05 0.00710  1.25093E-06 0.36295 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.79746E-05 0.02518  5.79609E-05 0.02521  1.55285E-06 0.49929 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.96127E-04 0.02279  1.96091E-04 0.02281  1.97457E-04 0.35157 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58116E+01 0.01114 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48257E+01 0.00027  3.93776E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 23:30:20 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00352E+00  9.99073E-01  1.00726E+00  1.00550E+00  1.00032E+00  1.00052E+00  9.97786E-01  1.00029E+00  9.99138E-01  1.00175E+00  9.95296E-01  9.98929E-01  9.98772E-01  9.98831E-01  9.97825E-01  1.00349E+00  9.93748E-01  9.97955E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.73632E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.62637E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36822E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50870E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64688E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48439E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47519E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27944E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.23203E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500588 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50051E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50051E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.76099E+03 ;
RUNNING_TIME              (idx, 1)        =  4.82641E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.93422E+00  1.62767E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.79139E+02  4.05890E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.22833E-01  1.88667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.81667E-02  1.36667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.18628E+00  5.80283E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.82106E+02  5.63574E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86445 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98498E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86869E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.91455E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74034E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.62649E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92265E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33693E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95779E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57273E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53993E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39205E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36522E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14199E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69295E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23937E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58199E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34686E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31698E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52912E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.98493E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59933E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23473E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03504E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.15305E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01684E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04509E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06363E-01 -2.34847E+28  2.44282E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01755E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.51590E+18 0.00403  1.61870E-02 0.00396 ];
U233_FISS                 (idx, [1:   4]) = [  7.93803E+19 0.00054  8.47693E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.15047E+18 0.00193  7.63574E-02 0.00184 ];
U238_FISS                 (idx, [1:   4]) = [  7.74659E+13 0.57593  8.27191E-07 0.57591 ];
PU239_FISS                (idx, [1:   4]) = [  6.16166E+17 0.00655  6.57979E-03 0.00652 ];
PU240_FISS                (idx, [1:   4]) = [  7.45418E+16 0.01787  7.95849E-04 0.01786 ];
PU241_FISS                (idx, [1:   4]) = [  1.28439E+17 0.01383  1.37157E-03 0.01381 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26811E+19 0.00058  6.44376E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15778E+19 0.00143  8.04978E-02 0.00139 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51971E+18 0.00302  1.75190E-02 0.00300 ];
U238_CAPT                 (idx, [1:   4]) = [  2.63750E+15 0.10078  1.83473E-05 0.10084 ];
PU239_CAPT                (idx, [1:   4]) = [  2.97136E+17 0.00946  2.06585E-03 0.00945 ];
PU240_CAPT                (idx, [1:   4]) = [  1.89699E+17 0.01230  1.31920E-03 0.01234 ];
PU241_CAPT                (idx, [1:   4]) = [  2.89052E+16 0.02969  2.01003E-04 0.02972 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80331E+17 0.01216  1.25349E-03 0.01210 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002037 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07265E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002037 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5500078 5.50502E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3581130 3.58418E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 920829 9.21526E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002037 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36591E+20 4.9E-06  2.36591E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35951E+19 1.1E-07  9.35951E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43842E+20 0.00026  1.28615E+20 0.00020  1.52273E+19 0.00169 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37437E+20 0.00016  2.22210E+20 0.00011  1.52273E+19 0.00169 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61271E+20 0.00036  2.61271E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49789E+22 0.00024  3.31677E+22 0.00023  1.81112E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40777E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61515E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17043E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.06063E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.06063E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.67350E-01 0.04578 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.25802E-01 0.01085 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.54664E-04 0.01976 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.55877E+03 0.02486 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07852E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 6.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.63730E-01 0.03554 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.02611E-01 0.03554 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52781E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00059E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05845E-01 0.00045  5.64336E-02 0.00044  1.94533E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05685E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05583E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05685E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97632E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85142E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85186E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58756E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58653E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07225E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.06803E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93740E-03 0.00529  2.78103E-04 0.01993  9.25654E-04 0.01084  7.12335E-04 0.01267  1.56647E-03 0.00852  3.69269E-04 0.01810  8.55717E-05 0.03578 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.17467E-01 0.01472  1.24175E-02 0.00355  3.22014E-02 0.00016  1.06746E-01 0.00058  3.01198E-01 0.00032  1.25743E+00 0.00102  5.94067E+00 0.02703 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48135E-03 0.00746  2.47630E-04 0.02751  8.05475E-04 0.01590  6.37677E-04 0.01838  1.38308E-03 0.01168  3.29217E-04 0.02598  7.82654E-05 0.05267 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.23613E-01 0.02138  1.24802E-02 0.00012  3.22075E-02 0.00023  1.06864E-01 0.00081  3.01185E-01 0.00045  1.25683E+00 0.00132  6.95066E+00 0.01991 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.60282E-07 0.00128  8.59863E-07 0.00128  9.81459E-07 0.02198 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.79228E-07 0.00122  7.78848E-07 0.00122  8.89228E-07 0.02207 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43845E-03 0.00902  2.42943E-04 0.03386  7.84506E-04 0.01924  6.26899E-04 0.02001  1.37266E-03 0.01418  3.31997E-04 0.02927  7.94481E-05 0.05786 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.34288E-01 0.02484  1.24783E-02 9.1E-05  3.21949E-02 0.00028  1.06735E-01 0.00105  3.01010E-01 0.00053  1.25166E+00 0.00227  7.15936E+00 0.02743 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.51763E-07 0.01709  7.51516E-07 0.01709  8.14700E-07 0.05559 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.81083E-07 0.01709  6.80860E-07 0.01709  7.38477E-07 0.05569 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14653E-03 0.04260  2.15843E-04 0.14832  6.40085E-04 0.08508  6.89201E-04 0.08897  1.19470E-03 0.05834  3.38161E-04 0.11249  6.85417E-05 0.22433 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.05786E-01 0.08322  1.24745E-02 0.00021  3.21945E-02 0.00099  1.06721E-01 0.00275  3.00940E-01 0.00186  1.25583E+00 0.00377  5.91939E+00 0.10089 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.12200E-03 0.04205  2.26561E-04 0.14488  6.27209E-04 0.08279  6.76687E-04 0.08926  1.20493E-03 0.05677  3.24098E-04 0.11229  6.25169E-05 0.21554 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.02725E-01 0.08475  1.24745E-02 0.00021  3.21935E-02 0.00099  1.06733E-01 0.00276  3.00981E-01 0.00186  1.25597E+00 0.00378  5.93844E+00 0.10116 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.20944E+03 0.03956 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.51834E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.71564E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40865E-03 0.00573 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.00251E+03 0.00579 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.01701E-09 0.00090 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77634E-05 0.00721  6.77506E-05 0.00721  2.46138E-06 0.26730 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.98712E-05 0.02055  5.98739E-05 0.02060  2.27707E-06 0.32077 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.47304E-04 0.01911  2.46871E-04 0.01913  3.58030E-04 0.26346 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58034E+01 0.01096 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47519E+01 0.00026  3.90609E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 00:11:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01099E+00  9.92943E-01  1.00307E+00  1.00217E+00  9.99371E-01  1.00338E+00  9.98548E-01  1.00862E+00  1.00070E+00  9.99580E-01  9.99038E-01  9.96026E-01  9.96843E-01  9.94138E-01  9.94792E-01  9.97104E-01  9.99953E-01  1.00274E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.79211E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.62079E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35417E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49639E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66133E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49065E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48136E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33578E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.28991E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500788 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50046E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50046E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.16411E+03 ;
RUNNING_TIME              (idx, 1)        =  5.23583E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.09447E+00  1.60250E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.19860E+02  4.07211E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.42417E-01  1.95833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.81667E-02  1.36667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.69043E+00  5.04100E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.23124E+02  5.63893E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86302 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98510E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86713E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.91108E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73824E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.41819E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91217E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33179E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96471E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57104E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64083E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56564E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45862E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.15995E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76463E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39438E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93105E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33844E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31075E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52044E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00084E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00819E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22431E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02294E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34643E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99912E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04736E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61102E-01 -3.55709E+28  2.56368E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01502E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.51568E+18 0.00397  1.61880E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  7.92716E+19 0.00054  8.46637E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.11917E+18 0.00201  7.60328E-02 0.00192 ];
U238_FISS                 (idx, [1:   4]) = [  1.58037E+14 0.40571  1.69033E-06 0.40572 ];
PU239_FISS                (idx, [1:   4]) = [  6.59915E+17 0.00618  7.04860E-03 0.00620 ];
PU240_FISS                (idx, [1:   4]) = [  8.50814E+16 0.01788  9.08609E-04 0.01788 ];
PU241_FISS                (idx, [1:   4]) = [  1.53610E+17 0.01262  1.64084E-03 0.01264 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22627E+19 0.00054  6.40090E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15637E+19 0.00148  8.02266E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50750E+18 0.00306  1.73971E-02 0.00307 ];
U238_CAPT                 (idx, [1:   4]) = [  3.14014E+15 0.09732  2.18183E-05 0.09734 ];
PU239_CAPT                (idx, [1:   4]) = [  3.12943E+17 0.00930  2.17127E-03 0.00930 ];
PU240_CAPT                (idx, [1:   4]) = [  2.28082E+17 0.01040  1.58218E-03 0.01036 ];
PU241_CAPT                (idx, [1:   4]) = [  3.28408E+16 0.02848  2.27811E-04 0.02848 ];
SM149_CAPT                (idx, [1:   4]) = [  1.89097E+17 0.01237  1.31211E-03 0.01239 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001840 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08856E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001840 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5499686 5.50492E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3572873 3.57592E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 929281 9.30045E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001840 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.44589E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36656E+20 4.7E-06  2.36656E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35888E+19 1.1E-07  9.35888E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44045E+20 0.00026  1.28364E+20 0.00018  1.56815E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37634E+20 0.00016  2.21952E+20 0.00010  1.56815E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61840E+20 0.00033  2.61840E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53136E+22 0.00023  3.34699E+22 0.00022  1.84364E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43533E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61987E+20 0.00018 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17461E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.88058E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.88058E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.05176E-01 0.04329 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21141E-01 0.00945 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.14072E-04 0.01902 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.30610E+03 0.02476 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07000E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.60178E-01 0.02794 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.89427E-01 0.02794 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52868E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04128E-01 0.00043  5.63122E-02 0.00042  1.99628E-04 0.00880 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.04298E-01 0.00018 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03860E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.04298E-01 0.00018 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97080E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85264E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84957E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58566E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59013E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.06974E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07864E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.97267E-03 0.00537  2.81235E-04 0.01992  9.26790E-04 0.01044  7.08040E-04 0.01294  1.60273E-03 0.00874  3.71396E-04 0.01739  8.24774E-05 0.03722 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.11584E-01 0.01437  1.24801E-02 9.8E-05  3.21931E-02 0.00016  1.06741E-01 0.00053  3.01338E-01 0.00031  1.25947E+00 0.00105  5.96971E+00 0.02782 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49769E-03 0.00775  2.56094E-04 0.02863  8.13453E-04 0.01659  6.05853E-04 0.01809  1.42256E-03 0.01200  3.26372E-04 0.02446  7.33603E-05 0.05095 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.12908E-01 0.01965  1.24808E-02 0.00016  3.21928E-02 0.00023  1.06768E-01 0.00076  3.01268E-01 0.00044  1.26087E+00 0.00133  7.09939E+00 0.01943 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.70526E-07 0.00138  8.70020E-07 0.00138  1.01364E-06 0.01842 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.87010E-07 0.00132  7.86553E-07 0.00132  9.16260E-07 0.01838 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.52764E-03 0.00891  2.53828E-04 0.03418  8.26557E-04 0.01801  6.15601E-04 0.02291  1.42566E-03 0.01449  3.35081E-04 0.02885  7.09153E-05 0.06729 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07682E-01 0.02587  1.24792E-02 0.00015  3.21931E-02 0.00029  1.06804E-01 0.00101  3.01438E-01 0.00054  1.25994E+00 0.00187  6.92043E+00 0.03120 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.60563E-07 0.01721  7.60212E-07 0.01721  7.92166E-07 0.05279 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.87842E-07 0.01721  6.87523E-07 0.01721  7.16307E-07 0.05285 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.03495E-03 0.04230  2.91555E-04 0.12528  7.29458E-04 0.07939  5.03015E-04 0.09198  1.16888E-03 0.06320  2.84133E-04 0.12926  5.79056E-05 0.26375 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.08248E-01 0.10623  1.24870E-02 0.00076  3.21625E-02 0.00090  1.06997E-01 0.00340  3.00811E-01 0.00182  1.25855E+00 0.00633  7.75213E+00 0.08338 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.01758E-03 0.04140  2.71461E-04 0.12294  7.33899E-04 0.07744  5.05403E-04 0.09070  1.16641E-03 0.06354  2.82337E-04 0.12595  5.80661E-05 0.25799 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.05343E-01 0.10287  1.24871E-02 0.00076  3.21658E-02 0.00090  1.07010E-01 0.00340  3.00906E-01 0.00183  1.25850E+00 0.00631  7.69512E+00 0.08316 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.01992E+03 0.03916 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60938E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78340E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.58171E-03 0.00597 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.16189E+03 0.00609 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16003E-09 0.00090 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.86901E-05 0.00582  6.86964E-05 0.00583  1.89926E-06 0.31343 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.04584E-05 0.02078  6.04710E-05 0.02083  1.85671E-06 0.40529 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.00687E-04 0.01870  3.00791E-04 0.01865  2.87133E-04 0.30005 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62555E+01 0.01082 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48136E+01 0.00028  3.90621E+01 0.00040 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:27:42 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 00:34:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170462419 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02571E+00  9.98622E-01  1.00021E+00  1.00235E+00  9.95845E-01  1.00742E+00  9.94545E-01  9.96106E-01  9.98452E-01  1.00105E+00  9.96766E-01  1.00223E+00  1.00212E+00  1.00093E+00  9.96263E-01  9.89611E-01  9.96361E-01  9.95414E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.85885E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61412E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.33916E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.48356E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67524E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49628E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48693E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.39345E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.35468E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499965 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50012E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50012E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.39272E+03 ;
RUNNING_TIME              (idx, 1)        =  5.47074E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.08155E+00  1.08155E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.25840E+00  1.63933E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.43126E+02  2.32664E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.61200E-01  1.87833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.81667E-02  1.36667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.20037E+00  5.09883E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.46610E+02  5.46610E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85739 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97630E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86146E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22625.53;
MEMSIZE                   (idx, 1)        = 22466.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 314.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.90884E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73787E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.64099E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90635E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32843E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96823E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57094E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65388E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61379E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.46940E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16110E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78581E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44109E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03151E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33713E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30986E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51913E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00513E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13481E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21994E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02066E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.38396E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98893E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04785E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16253E-01 -4.77480E+28  2.68545E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01152E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.50292E+18 0.00401  1.60590E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  7.91896E+19 0.00057  8.46213E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.11901E+18 0.00195  7.60745E-02 0.00190 ];
U238_FISS                 (idx, [1:   4]) = [  2.09741E+14 0.35045  2.23962E-06 0.35046 ];
PU239_FISS                (idx, [1:   4]) = [  6.63017E+17 0.00669  7.08388E-03 0.00661 ];
PU240_FISS                (idx, [1:   4]) = [  8.84942E+16 0.01742  9.45603E-04 0.01740 ];
PU241_FISS                (idx, [1:   4]) = [  1.59603E+17 0.01230  1.70548E-03 0.01228 ];
TH232_CAPT                (idx, [1:   4]) = [  9.18401E+19 0.00054  6.37144E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15397E+19 0.00147  8.00572E-02 0.00142 ];
U235_CAPT                 (idx, [1:   4]) = [  2.52126E+18 0.00351  1.74905E-02 0.00345 ];
U238_CAPT                 (idx, [1:   4]) = [  3.06487E+15 0.09101  2.12822E-05 0.09110 ];
PU239_CAPT                (idx, [1:   4]) = [  3.13864E+17 0.00923  2.17747E-03 0.00923 ];
PU240_CAPT                (idx, [1:   4]) = [  2.30778E+17 0.01048  1.60098E-03 0.01045 ];
PU241_CAPT                (idx, [1:   4]) = [  3.38919E+16 0.02749  2.35029E-04 0.02747 ];
SM149_CAPT                (idx, [1:   4]) = [  1.91355E+17 0.01195  1.32759E-03 0.01195 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000463 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08189E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000463 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496757 5.50249E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3568622 3.57233E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 935084 9.36008E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000463 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.65775E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36667E+20 4.7E-06  2.36667E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35882E+19 1.0E-07  9.35882E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44091E+20 0.00030  1.28052E+20 0.00019  1.60383E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37679E+20 0.00018  2.21641E+20 0.00011  1.60383E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61963E+20 0.00033  2.61963E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.55960E+22 0.00026  3.37344E+22 0.00021  1.86153E+21 0.00193 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.45205E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62199E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17661E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.71993E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.71993E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.05237E-01 0.03798 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06768E-01 0.00864 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.76948E-04 0.01665 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.74124E+03 0.02629 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06404E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.04721E-01 0.02445 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.29413E-01 0.02445 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52881E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00074E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03369E-01 0.00045  5.62672E-02 0.00045  1.94288E-04 0.00827 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.03609E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03473E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.03609E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96940E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85136E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85226E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58770E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58583E-02 0.00085 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08137E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07947E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94751E-03 0.00546  2.69729E-04 0.02000  9.29854E-04 0.01115  7.12217E-04 0.01200  1.58986E-03 0.00821  3.61229E-04 0.01785  8.46225E-05 0.03650 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.15007E-01 0.01513  1.24166E-02 0.00355  3.21965E-02 0.00016  1.06836E-01 0.00053  3.01495E-01 0.00030  1.25816E+00 0.00114  5.84676E+00 0.02885 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48052E-03 0.00756  2.34894E-04 0.02882  8.17529E-04 0.01655  6.29375E-04 0.01850  1.39895E-03 0.01149  3.22791E-04 0.02604  7.69760E-05 0.05099 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.19030E-01 0.02001  1.24782E-02 7.9E-05  3.21969E-02 0.00022  1.06792E-01 0.00072  3.01727E-01 0.00043  1.25657E+00 0.00150  6.95611E+00 0.02058 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80696E-07 0.00151  8.80340E-07 0.00151  9.78575E-07 0.01623 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.95512E-07 0.00141  7.95190E-07 0.00141  8.84073E-07 0.01626 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44046E-03 0.00818  2.30302E-04 0.03411  8.32334E-04 0.01853  6.21841E-04 0.02128  1.38008E-03 0.01391  2.99976E-04 0.03061  7.59306E-05 0.06358 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.16093E-01 0.02673  1.24801E-02 0.00011  3.22069E-02 0.00029  1.06770E-01 0.00097  3.01329E-01 0.00053  1.26291E+00 0.00175  7.06312E+00 0.02978 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.67182E-07 0.01721  7.66723E-07 0.01721  8.27136E-07 0.05553 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.93271E-07 0.01720  6.92857E-07 0.01720  7.47594E-07 0.05551 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06959E-03 0.04279  2.23867E-04 0.14546  6.84548E-04 0.08926  5.66821E-04 0.09254  1.23875E-03 0.06145  2.76360E-04 0.12833  7.92412E-05 0.21671 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.84554E-01 0.10696  1.24833E-02 0.00095  3.21610E-02 0.00093  1.07343E-01 0.00360  3.01046E-01 0.00189  1.25183E+00 0.00440  7.71609E+00 0.06669 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.05962E-03 0.04195  2.21906E-04 0.14558  6.73177E-04 0.08786  5.63491E-04 0.09163  1.24394E-03 0.05975  2.81958E-04 0.12767  7.51486E-05 0.21481 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.84059E-01 0.10639  1.24835E-02 0.00095  3.21648E-02 0.00092  1.07345E-01 0.00359  3.01047E-01 0.00187  1.25154E+00 0.00430  7.71367E+00 0.06671 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.03913E+03 0.04000 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.71410E-07 0.00084 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.87136E-07 0.00068 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38800E-03 0.00592 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.88821E+03 0.00588 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.31057E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.94728E-05 0.00534  6.94741E-05 0.00535  1.22502E-06 0.38265 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.15902E-05 0.01625  6.14572E-05 0.01620  2.20236E-06 0.41139 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.60330E-04 0.01636  3.61056E-04 0.01640  1.87738E-04 0.37717 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.56441E+01 0.01122 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48693E+01 0.00027  3.90548E+01 0.00038 ];

