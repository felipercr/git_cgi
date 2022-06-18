
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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 17:54:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03513E+00  1.01179E+00  9.93073E-01  9.90232E-01  1.00078E+00  9.92186E-01  1.00085E+00  1.00162E+00  9.93063E-01  9.97706E-01  9.93646E-01  9.91691E-01  9.94400E-01  1.01221E+00  9.95448E-01  9.96898E-01  1.00166E+00  9.97618E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.79591E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72041E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40945E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54755E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60096E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.44485E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.43579E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.06808E+01 0.00034  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.87182E-01 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333478 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33394E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33394E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.54478E+02 ;
RUNNING_TIME              (idx, 1)        =  2.82025E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.81167E-01  1.81167E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.70543E+01  2.70543E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.03135E+00  1.98307E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.62596E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.02326 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98593E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.95112E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  2.83473E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.22928E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13057E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.83473E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.22928E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.72448E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.44627E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.72448E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.44627E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29430E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.12960E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.30559E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.10835E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.71402E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.53834E+18 0.00363  1.63877E-02 0.00357 ];
U233_FISS                 (idx, [1:   4]) = [  9.23289E+19 0.00048  9.83612E-01 5.9E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21403E+19 0.00052  7.56136E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34060E+19 0.00134  1.10014E-01 0.00126 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001821 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15774E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001821 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5137644 5.14286E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3957841 3.96160E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906336 9.07125E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001821 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.16067E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36467E+20 3.7E-06  2.36467E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.2E-07  9.37766E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21845E+20 0.00033  1.08623E+20 0.00025  1.32227E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15622E+20 0.00019  2.02399E+20 0.00013  1.32227E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36945E+20 0.00037  2.36945E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.11469E+22 0.00027  2.95287E+22 0.00024  1.61822E+21 0.00189 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.14944E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.37116E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05220E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51104E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.38093E-01 0.08481 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.74102E-01 0.00920 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.21536E-04 0.01725 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.20081E+04 0.02151 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09295E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 9.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.64930E-01 0.06750 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.22734E-01 0.06750 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52160E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.98952E-01 0.00042  1.10643E-01 0.00040  3.51898E-04 0.00889 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.98427E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.98022E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.98427E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09808E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00138E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00195E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36629E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36534E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49863E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49737E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.22076E-03 0.00548  2.65948E-04 0.02034  7.99265E-04 0.01205  5.79502E-04 0.01259  1.26206E-03 0.00911  2.65985E-04 0.01903  4.80060E-05 0.04531 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43539E-01 0.01463  1.24753E-02 2.9E-05  3.23326E-02 8.7E-05  1.06063E-01 0.00059  2.97201E-01 0.00026  1.23565E+00 0.00022  5.21003E+00 0.04032 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19822E-03 0.00859  2.64635E-04 0.03012  7.95484E-04 0.01744  5.77546E-04 0.01991  1.24369E-03 0.01373  2.69622E-04 0.02770  4.72346E-05 0.06084 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45534E-01 0.01997  1.24754E-02 4.1E-05  3.23275E-02 0.00012  1.06088E-01 0.00084  2.97214E-01 0.00037  1.23599E+00 0.00030  6.67390E+00 0.02898 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.03252E-07 0.00123  8.02986E-07 0.00124  8.82153E-07 0.01875 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.02373E-07 0.00118  8.02108E-07 0.00118  8.81058E-07 0.01869 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.16663E-03 0.00895  2.60454E-04 0.03159  7.92467E-04 0.01865  5.68965E-04 0.02136  1.24529E-03 0.01400  2.51948E-04 0.02914  4.75116E-05 0.07225 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.37729E-01 0.02313  1.24758E-02 5.3E-05  3.23324E-02 0.00015  1.06211E-01 0.00105  2.97261E-01 0.00043  1.23599E+00 0.00039  6.31612E+00 0.04195 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.19744E-07 0.01972  7.19598E-07 0.01972  7.61778E-07 0.05701 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.19123E-07 0.01971  7.18977E-07 0.01972  7.61144E-07 0.05706 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.67450E-03 0.04375  2.71119E-04 0.11930  6.72422E-04 0.08151  4.37504E-04 0.09726  1.03943E-03 0.06606  2.24920E-04 0.13940  2.91040E-05 0.31804 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.67212E-01 0.11517  1.24763E-02 0.00012  3.23826E-02 0.00070  1.06108E-01 0.00328  2.97090E-01 0.00152  1.23526E+00 0.00119  7.91215E+00 0.11040 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.68739E-03 0.04310  2.67724E-04 0.11771  6.92571E-04 0.08191  4.37985E-04 0.09600  1.04110E-03 0.06418  2.18915E-04 0.13477  2.90973E-05 0.31328 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.66595E-01 0.12040  1.24763E-02 0.00012  3.23832E-02 0.00070  1.06070E-01 0.00322  2.97095E-01 0.00150  1.23531E+00 0.00119  7.91215E+00 0.11040 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.73392E+03 0.03971 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  7.97452E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.96572E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17137E-03 0.00566 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.97714E+03 0.00566 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.04340E-09 0.00129 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.37308E-05 0.00477  7.37548E-05 0.00478  2.86639E-06 0.28074 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.63749E-05 0.01614  7.62327E-05 0.01615  4.57805E-06 0.51962 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.16421E-04 0.01655  4.16458E-04 0.01661  4.23233E-04 0.27257 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74472E+01 0.01118 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.43579E+01 0.00028  3.99640E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 18:38:22 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01333E+00  1.00657E+00  9.99619E-01  9.92489E-01  1.00242E+00  1.00110E+00  1.00212E+00  1.00276E+00  9.98963E-01  9.93586E-01  9.99384E-01  9.95060E-01  9.97166E-01  1.00329E+00  9.99771E-01  9.97528E-01  9.95305E-01  9.99536E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.78957E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72104E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.41013E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54793E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60363E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45296E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44387E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.08146E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.86385E-01 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333552 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33373E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33373E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.58823E+02 ;
RUNNING_TIME              (idx, 1)        =  7.15995E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.95700E-01  1.14533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.03129E+01  4.32585E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.76833E-02  1.76833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.21230E+00  3.18090E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.84597E+01  4.21885E+02 ];
CPU_USAGE                 (idx, 1)        = 9.20150 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99000E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.13636E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  5.77077E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60377E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65590E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.47472E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.03856E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26539E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46917E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.46053E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65544E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.39729E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.77002E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06323E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17844E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44265E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35697E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17608E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55499E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60705E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38176E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39384E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51318E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.52481E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.29417E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.17287E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.07986E-04 -2.38429E+25  2.20820E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.83040E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.53562E+18 0.00392  1.63855E-02 0.00386 ];
U233_FISS                 (idx, [1:   4]) = [  9.21530E+19 0.00050  9.83321E-01 6.4E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.86861E+14 0.30633  3.06462E-06 0.30598 ];
TH232_CAPT                (idx, [1:   4]) = [  9.31556E+19 0.00053  7.51820E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34042E+19 0.00132  1.08181E-01 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  9.69379E+13 0.49750  7.78341E-07 0.49749 ];
SM149_CAPT                (idx, [1:   4]) = [  1.21653E+16 0.04371  9.81612E-05 0.04367 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001200 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15074E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001200 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5176665 5.18233E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3915711 3.91963E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908824 9.09544E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001200 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.63685E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36458E+20 3.9E-06  2.36458E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37762E+19 1.2E-07  9.37762E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23903E+20 0.00030  1.10518E+20 0.00024  1.33847E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17679E+20 0.00017  2.04294E+20 0.00013  1.33847E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.39096E+20 0.00036  2.39096E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.14687E+22 0.00026  2.98328E+22 0.00024  1.63581E+21 0.00161 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.17477E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39427E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06368E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51194E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51194E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.83196E-01 0.08125 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.04742E-01 0.00931 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.51055E-04 0.01745 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.12066E+04 0.02587 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09052E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.78542E-01 0.06525 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.34987E-01 0.06525 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52151E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.88282E-01 0.00041  1.09460E-01 0.00040  3.57875E-04 0.00838 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.88757E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.89006E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.88757E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08766E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00672E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00510E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35898E-02 0.00124 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36104E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49947E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50012E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30483E-03 0.00573  2.64420E-04 0.01816  8.07764E-04 0.01129  6.07040E-04 0.01244  1.30587E-03 0.00876  2.67942E-04 0.02112  5.17946E-05 0.04460 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43015E-01 0.01389  1.24758E-02 3.0E-05  3.23370E-02 9.2E-05  1.06015E-01 0.00056  2.97306E-01 0.00027  1.23546E+00 0.00020  5.09500E+00 0.03929 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23286E-03 0.00769  2.55218E-04 0.02669  7.83128E-04 0.01646  6.02503E-04 0.01950  1.27540E-03 0.01238  2.67118E-04 0.02876  4.94957E-05 0.06610 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42082E-01 0.02124  1.24758E-02 3.8E-05  3.23327E-02 0.00012  1.06024E-01 0.00082  2.97355E-01 0.00037  1.23550E+00 0.00030  6.18848E+00 0.03023 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.14215E-07 0.00115  8.13964E-07 0.00115  8.97261E-07 0.01675 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.04630E-07 0.00105  8.04383E-07 0.00106  8.86746E-07 0.01676 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26459E-03 0.00862  2.57583E-04 0.03087  7.86968E-04 0.01861  5.87997E-04 0.02077  1.30222E-03 0.01315  2.73051E-04 0.03312  5.67784E-05 0.06587 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.56906E-01 0.02289  1.24756E-02 5.3E-05  3.23322E-02 0.00015  1.06022E-01 0.00102  2.97293E-01 0.00042  1.23621E+00 0.00036  6.24619E+00 0.03866 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.21597E-07 0.01955  7.21245E-07 0.01956  8.44518E-07 0.06054 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.13363E-07 0.01956  7.13015E-07 0.01956  8.34409E-07 0.06040 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.07171E-03 0.04178  3.10062E-04 0.12799  7.51121E-04 0.07941  4.63569E-04 0.09722  1.26972E-03 0.05652  2.39016E-04 0.13932  3.82294E-05 0.23195 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.09344E-01 0.06806  1.24758E-02 0.00013  3.23192E-02 0.00042  1.06481E-01 0.00371  2.96236E-01 0.00124  1.23596E+00 0.00106  5.02331E+00 0.12462 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06686E-03 0.04109  3.06803E-04 0.12472  7.55056E-04 0.07803  4.64889E-04 0.09547  1.26617E-03 0.05523  2.36486E-04 0.13638  3.74601E-05 0.22631 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.06539E-01 0.06660  1.24758E-02 0.00013  3.23174E-02 0.00041  1.06472E-01 0.00368  2.96204E-01 0.00122  1.23609E+00 0.00105  5.02331E+00 0.12462 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.25932E+03 0.03685 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.08953E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.99436E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.31918E-03 0.00590 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.10369E+03 0.00595 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.97669E-09 0.00116 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.20967E-05 0.00560  7.20488E-05 0.00559  2.90280E-06 0.32130 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.83334E-05 0.01814  7.83992E-05 0.01818  1.50320E-06 0.41945 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.45186E-04 0.01674  3.45225E-04 0.01670  3.30745E-04 0.31787 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65788E+01 0.01135 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44387E+01 0.00025  4.00574E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 19:24:35 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01574E+00  1.00025E+00  9.97828E-01  9.97465E-01  9.96583E-01  1.00621E+00  1.00946E+00  9.95167E-01  1.00951E+00  9.89243E-01  9.97646E-01  1.00104E+00  9.99817E-01  1.00412E+00  9.92864E-01  1.00165E+00  9.88719E-01  9.96676E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.77656E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72234E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.41166E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54882E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59821E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46454E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45545E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.09934E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.83324E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333920 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33409E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33409E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.08920E+03 ;
RUNNING_TIME              (idx, 1)        =  1.17826E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.14317E-01  1.18617E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.16392E+02  4.60796E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.66667E-02  1.89833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35667E-02  1.35667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.63712E+00  3.42477E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.14440E+02  6.35194E+02 ];
CPU_USAGE                 (idx, 1)        = 9.24414 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99045E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.18043E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.38706E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65407E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.22452E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85829E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29842E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49786E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49325E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.76233E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39972E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.86483E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.98335E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89728E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60138E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49508E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34661E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33444E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54399E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.73198E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17959E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.00949E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51104E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.32707E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.51111E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.26644E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45045E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.74863E-04 -1.26928E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.99365E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.57087E+18 0.00397  1.67497E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  9.20904E+19 0.00047  9.81926E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.06350E+16 0.04665  1.13381E-04 0.04663 ];
TH232_CAPT                (idx, [1:   4]) = [  9.45548E+19 0.00055  7.46412E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33830E+19 0.00138  1.05646E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  3.96804E+15 0.07680  3.13343E-05 0.07681 ];
SM149_CAPT                (idx, [1:   4]) = [  5.76370E+16 0.02219  4.55023E-04 0.02220 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002271 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12902E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002271 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5225119 5.23004E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3868688 3.87204E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908464 9.09211E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002271 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36447E+20 3.6E-06  2.36447E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37757E+19 1.2E-07  9.37757E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26678E+20 0.00028  1.13080E+20 0.00022  1.35976E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20454E+20 0.00016  2.06856E+20 0.00012  1.35976E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42215E+20 0.00036  2.42215E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.19127E+22 0.00025  3.02557E+22 0.00024  1.65701E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20232E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42477E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08033E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51430E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51430E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.40850E-01 0.05742 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13720E-01 0.01112 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.29328E-04 0.02095 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.02811E+04 0.02549 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09083E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.90851E-01 0.04306 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.28066E-01 0.04306 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52141E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99674E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.76194E-01 0.00041  1.08125E-01 0.00040  3.50991E-04 0.00871 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.76263E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.76226E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.76263E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07391E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01101E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01074E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35316E-02 0.00125 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35339E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50842E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50825E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.35552E-03 0.00526  2.70076E-04 0.01904  8.17294E-04 0.01026  6.16624E-04 0.01338  1.33570E-03 0.00828  2.63506E-04 0.02036  5.23152E-05 0.04474 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.39974E-01 0.01465  1.24763E-02 2.5E-05  3.23303E-02 8.6E-05  1.06135E-01 0.00061  2.97412E-01 0.00028  1.23520E+00 0.00024  5.10079E+00 0.03940 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24871E-03 0.00753  2.52384E-04 0.02734  8.07864E-04 0.01501  5.95867E-04 0.01796  1.28053E-03 0.01251  2.57597E-04 0.02794  5.44589E-05 0.06609 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.52663E-01 0.02454  1.24766E-02 3.4E-05  3.23246E-02 0.00011  1.06093E-01 0.00079  2.97289E-01 0.00037  1.23545E+00 0.00032  6.30329E+00 0.03044 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.25125E-07 0.00108  8.24846E-07 0.00109  9.10076E-07 0.01463 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.05449E-07 0.00103  8.05176E-07 0.00104  8.88408E-07 0.01463 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23566E-03 0.00892  2.51106E-04 0.03232  7.87489E-04 0.01628  6.03590E-04 0.02171  1.29322E-03 0.01367  2.48146E-04 0.03446  5.21120E-05 0.06844 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.37857E-01 0.02361  1.24761E-02 4.8E-05  3.23334E-02 0.00015  1.06075E-01 0.00098  2.97204E-01 0.00040  1.23531E+00 0.00047  6.05309E+00 0.04242 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.30339E-07 0.01957  7.30110E-07 0.01958  7.74610E-07 0.05262 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.13174E-07 0.01957  7.12952E-07 0.01957  7.56136E-07 0.05249 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.75734E-03 0.04425  2.21107E-04 0.13574  7.08654E-04 0.08178  4.88932E-04 0.09556  1.14297E-03 0.06060  1.35120E-04 0.15411  6.05585E-05 0.24143 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.46007E-01 0.09789  1.24760E-02 0.00014  3.23232E-02 0.00045  1.06341E-01 0.00355  2.97213E-01 0.00151  1.23427E+00 0.00141  5.78596E+00 0.11741 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.78862E-03 0.04341  2.25857E-04 0.13783  7.07847E-04 0.07944  4.95808E-04 0.09706  1.16148E-03 0.05973  1.38839E-04 0.15576  5.87872E-05 0.24061 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.38882E-01 0.09413  1.24760E-02 0.00014  3.23223E-02 0.00045  1.06351E-01 0.00356  2.97186E-01 0.00149  1.23424E+00 0.00141  5.78596E+00 0.11741 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79355E+03 0.03992 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.19531E-07 0.00062 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.99986E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21297E-03 0.00522 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.92087E+03 0.00525 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.82671E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.95832E-05 0.00675  6.95148E-05 0.00683  2.17029E-06 0.35483 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.78937E-05 0.02230  6.79076E-05 0.02231  1.54576E-06 0.52895 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.26074E-04 0.02027  2.26035E-04 0.02041  2.45347E-04 0.35025 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71124E+01 0.01127 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45545E+01 0.00028  4.01760E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 20:11:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01304E+00  9.98581E-01  9.97106E-01  9.98189E-01  1.00244E+00  9.95063E-01  9.98586E-01  1.00678E+00  9.99918E-01  9.96362E-01  9.99335E-01  1.00424E+00  1.00223E+00  9.94618E-01  9.99462E-01  9.95872E-01  9.98571E-01  9.99609E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76896E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72310E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.41190E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54874E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59521E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47775E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46872E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.12372E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.81830E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333709 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33358E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33358E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.52548E+03 ;
RUNNING_TIME              (idx, 1)        =  1.64655E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.37117E-01  1.22800E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.63070E+02  4.66779E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.43167E-02  1.76500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35667E-02  1.35667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.20735E+01  3.43632E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.61259E+02  6.69679E+02 ];
CPU_USAGE                 (idx, 1)        = 9.26473 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99056E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.20179E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.50391E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66988E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.09724E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89544E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32349E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57722E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50623E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.81552E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.56121E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.01894E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.17593E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79549E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74359E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48830E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36340E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34734E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56134E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.40518E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42596E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08237E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53973E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.12397E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.67161E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.32133E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12512E-03 -2.48422E+26  2.21045E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.11982E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.56932E+18 0.00399  1.67345E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  9.19515E+19 0.00051  9.80556E-01 7.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.83855E+16 0.02560  4.09418E-04 0.02562 ];
TH232_CAPT                (idx, [1:   4]) = [  9.54813E+19 0.00051  7.43182E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33839E+19 0.00138  1.04175E-01 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  1.41766E+16 0.04055  1.10358E-04 0.04055 ];
SM149_CAPT                (idx, [1:   4]) = [  9.53189E+16 0.01526  7.41997E-04 0.01527 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000750 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15555E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000750 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5258647 5.26450E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3838524 3.84256E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 903579 9.04497E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000750 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -8.00937E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36440E+20 3.8E-06  2.36440E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37750E+19 1.2E-07  9.37750E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28567E+20 0.00029  1.14812E+20 0.00021  1.37550E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22342E+20 0.00017  2.08587E+20 0.00011  1.37550E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.44044E+20 0.00034  2.44044E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.22101E+22 0.00025  3.05367E+22 0.00023  1.67331E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20745E+19 0.00155 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44416E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09176E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51433E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51433E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.31286E-01 0.05098 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27355E-01 0.01144 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90305E-04 0.02269 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.41577E+03 0.02690 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09553E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.63534E-01 0.03637 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.94517E-01 0.03637 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52136E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.68633E-01 0.00044  1.07305E-01 0.00043  3.46272E-04 0.00875 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.68492E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.68875E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.68492E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06476E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01447E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01513E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34853E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34748E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51915E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51735E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.36545E-03 0.00573  2.75588E-04 0.01941  8.25076E-04 0.01060  6.17599E-04 0.01353  1.31783E-03 0.00948  2.76726E-04 0.02009  5.26249E-05 0.04697 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.44924E-01 0.01574  1.24753E-02 2.8E-05  3.23361E-02 8.7E-05  1.05980E-01 0.00055  2.97306E-01 0.00026  1.23605E+00 0.00023  5.09184E+00 0.03898 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22580E-03 0.00826  2.64670E-04 0.02910  7.89560E-04 0.01666  5.86278E-04 0.01813  1.26287E-03 0.01377  2.68013E-04 0.02967  5.44094E-05 0.06691 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.55140E-01 0.02477  1.24751E-02 3.9E-05  3.23366E-02 0.00013  1.06076E-01 0.00082  2.97281E-01 0.00035  1.23581E+00 0.00032  6.18490E+00 0.03041 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.35348E-07 0.00128  8.35008E-07 0.00129  9.43729E-07 0.02054 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.09098E-07 0.00120  8.08769E-07 0.00121  9.14050E-07 0.02054 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20852E-03 0.00877  2.64606E-04 0.03237  7.81090E-04 0.01717  5.74724E-04 0.02143  1.27761E-03 0.01428  2.61447E-04 0.03119  4.90446E-05 0.07312 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51867E-01 0.02450  1.24750E-02 5.8E-05  3.23436E-02 0.00017  1.06031E-01 0.00097  2.97290E-01 0.00039  1.23659E+00 0.00040  6.85416E+00 0.03975 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.40489E-07 0.01960  7.40305E-07 0.01960  7.51620E-07 0.05473 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.17267E-07 0.01959  7.17090E-07 0.01959  7.28018E-07 0.05470 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92401E-03 0.04526  2.63929E-04 0.12532  6.96164E-04 0.07871  5.35358E-04 0.10026  1.16866E-03 0.06264  2.12203E-04 0.13248  4.76884E-05 0.35980 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.53165E-01 0.09710  1.24766E-02 0.00012  3.23223E-02 0.00045  1.06786E-01 0.00410  2.97167E-01 0.00157  1.23768E+00 0.00117  6.75662E+00 0.14230 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93829E-03 0.04408  2.60877E-04 0.12269  7.18344E-04 0.07788  5.48986E-04 0.09846  1.15919E-03 0.05963  2.06641E-04 0.12916  4.42565E-05 0.34220 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.42088E-01 0.09134  1.24765E-02 0.00012  3.23166E-02 0.00041  1.06702E-01 0.00399  2.97094E-01 0.00154  1.23770E+00 0.00117  6.75662E+00 0.14230 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96913E+03 0.04135 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.30665E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.04565E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23820E-03 0.00520 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.89828E+03 0.00513 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78489E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.85056E-05 0.00756  6.85006E-05 0.00766  2.32140E-06 0.33402 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.46909E-05 0.02806  6.45951E-05 0.02818  3.16544E-06 0.42757 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.87272E-04 0.02166  1.87009E-04 0.02174  2.63606E-04 0.32950 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75292E+01 0.01177 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46872E+01 0.00026  4.02958E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 21:00:01 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01603E+00  1.00049E+00  9.96591E-01  1.00542E+00  1.00170E+00  1.00222E+00  9.98419E-01  9.99580E-01  9.99286E-01  9.90868E-01  9.98581E-01  9.98752E-01  1.00179E+00  1.00413E+00  9.98140E-01  1.00137E+00  9.92980E-01  9.93666E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76443E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72356E-01 8.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.41159E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54826E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59056E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48777E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47867E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.14353E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.82229E-01 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334435 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33417E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33417E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.97817E+03 ;
RUNNING_TIME              (idx, 1)        =  2.13258E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.63100E-01  1.25983E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.11516E+02  4.84460E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.25333E-02  1.82167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35667E-02  1.35667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.56529E+01  3.57933E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.09719E+02  6.77940E+02 ];
CPU_USAGE                 (idx, 1)        = 9.27594 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99068E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.21342E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.53240E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66507E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.46275E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91788E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33757E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58300E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49975E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.99507E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.68273E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.28847E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.27717E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.70096E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85490E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93710E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32895E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32091E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52496E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.36423E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81540E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08113E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51393E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.48982E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.72009E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.42356E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.28131E-03 -5.03706E+26  2.21300E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.34881E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.59819E+18 0.00388  1.70366E-02 0.00382 ];
U233_FISS                 (idx, [1:   4]) = [  9.16400E+19 0.00051  9.76905E-01 7.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.56180E+17 0.01266  1.66476E-03 0.01262 ];
TH232_CAPT                (idx, [1:   4]) = [  9.71244E+19 0.00051  7.39150E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33716E+19 0.00135  1.01763E-01 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  5.23627E+16 0.02055  3.98582E-04 0.02059 ];
SM149_CAPT                (idx, [1:   4]) = [  1.33971E+17 0.01403  1.01955E-03 0.01403 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002518 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15876E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002518 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5305222 5.31015E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3787560 3.79093E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909736 9.10507E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002518 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.04891E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36429E+20 3.7E-06  2.36429E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37730E+19 1.1E-07  9.37730E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31386E+20 0.00033  1.17480E+20 0.00023  1.39059E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.25159E+20 0.00019  2.11253E+20 0.00013  1.39059E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47452E+20 0.00036  2.47452E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.27032E+22 0.00026  3.10137E+22 0.00023  1.68949E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.25314E+19 0.00155 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47691E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10947E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51158E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51158E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.80010E-01 0.05707 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.14146E-01 0.01484 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.61175E-04 0.02623 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.44387E+03 0.02860 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08952E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.80013E-01 0.04275 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.18247E-01 0.04275 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52129E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.55813E-01 0.00043  1.05858E-01 0.00042  3.39628E-04 0.00861 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55643E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.55490E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55643E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05139E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01417E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01439E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34893E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34845E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53847E-01 0.00166 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54161E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38626E-03 0.00555  2.73953E-04 0.02012  8.19037E-04 0.01088  6.26168E-04 0.01303  1.33210E-03 0.00847  2.81932E-04 0.01892  5.30671E-05 0.04294 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50133E-01 0.01413  1.24751E-02 3.2E-05  3.23319E-02 0.00010  1.06102E-01 0.00062  2.97611E-01 0.00028  1.23649E+00 0.00028  5.39144E+00 0.03769 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24552E-03 0.00771  2.63369E-04 0.02734  7.85395E-04 0.01610  5.91892E-04 0.02045  1.27793E-03 0.01300  2.69699E-04 0.02720  5.72320E-05 0.06437 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.67231E-01 0.02258  1.24752E-02 3.9E-05  3.23299E-02 0.00015  1.06106E-01 0.00083  2.97609E-01 0.00041  1.23647E+00 0.00033  6.60427E+00 0.02838 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.46735E-07 0.00131  8.46424E-07 0.00132  9.48063E-07 0.01767 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.09264E-07 0.00120  8.08966E-07 0.00120  9.06302E-07 0.01772 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19813E-03 0.00856  2.57742E-04 0.03179  7.89779E-04 0.01845  5.69620E-04 0.02200  1.26401E-03 0.01414  2.64007E-04 0.03171  5.29697E-05 0.07128 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.60936E-01 0.02542  1.24757E-02 5.1E-05  3.23305E-02 0.00017  1.05978E-01 0.00099  2.97684E-01 0.00049  1.23718E+00 0.00058  6.90006E+00 0.03873 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.50153E-07 0.01964  7.49907E-07 0.01965  8.18742E-07 0.05853 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.17250E-07 0.01964  7.17015E-07 0.01964  7.82451E-07 0.05851 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92786E-03 0.04513  2.79652E-04 0.13325  7.70930E-04 0.08235  4.41315E-04 0.11245  1.10236E-03 0.06670  2.81277E-04 0.13489  5.23231E-05 0.29570 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.64894E-01 0.09804  1.24773E-02 0.00012  3.23161E-02 0.00046  1.06301E-01 0.00382  2.97368E-01 0.00159  1.23712E+00 0.00102  6.17885E+00 0.11535 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93512E-03 0.04477  2.72610E-04 0.13159  7.65522E-04 0.08062  4.46491E-04 0.11332  1.11571E-03 0.06649  2.79969E-04 0.13313  5.48205E-05 0.27271 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.75615E-01 0.10115  1.24773E-02 0.00012  3.23121E-02 0.00043  1.06317E-01 0.00380  2.97452E-01 0.00162  1.23722E+00 0.00100  6.17885E+00 0.11535 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.91473E+03 0.04056 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.41405E-07 0.00081 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.04174E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.16096E-03 0.00567 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.75718E+03 0.00568 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76711E-09 0.00085 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68178E-05 0.01021  6.69098E-05 0.01031  6.90973E-07 0.53314 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.10119E-05 0.02932  6.10627E-05 0.02927  4.72103E-07 0.61808 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.61619E-04 0.02557  1.61769E-04 0.02568  1.09422E-04 0.50039 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70401E+01 0.01176 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47867E+01 0.00027  4.03732E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 21:48:53 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01465E+00  1.00544E+00  1.00315E+00  9.87971E-01  9.97721E-01  1.00191E+00  9.96276E-01  9.95340E-01  1.00309E+00  9.96305E-01  9.98328E-01  9.98397E-01  1.00404E+00  9.97089E-01  1.00132E+00  9.96780E-01  9.98377E-01  1.00382E+00  ];
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

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.1E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.75963E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72404E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.41243E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54883E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58728E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50569E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49664E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17417E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.83737E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334452 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33417E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33417E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.42216E+03 ;
RUNNING_TIME              (idx, 1)        =  2.62114E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.92067E-01  1.28967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.60208E+02  4.86913E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.12833E-02  1.87500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35667E-02  1.35667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.04667E+01  4.81377E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.57341E+02  6.94685E+02 ];
CPU_USAGE                 (idx, 1)        = 9.24084 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99010E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.17427E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.59945E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67448E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.89980E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95228E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36028E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61524E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50649E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09620E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79266E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.69074E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.43515E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25670E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94886E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35263E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33886E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32824E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53485E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.04214E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17431E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11975E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54575E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.28916E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.81958E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.49773E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85969E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.42097E-03 -7.55339E+26  2.21552E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.54736E-01 0.00079 ];
TH232_FISS                (idx, [1:   4]) = [  1.60996E+18 0.00406  1.71697E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  9.12177E+19 0.00057  9.72784E-01 9.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.32388E+17 0.00847  3.54491E-03 0.00847 ];
TH232_CAPT                (idx, [1:   4]) = [  9.83916E+19 0.00052  7.35402E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33152E+19 0.00144  9.95207E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  1.14326E+17 0.01499  8.54450E-04 0.01496 ];
SM149_CAPT                (idx, [1:   4]) = [  1.53539E+17 0.01278  1.14753E-03 0.01276 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002501 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14349E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002501 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5348427 5.35340E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3748709 3.75195E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905365 9.06092E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002501 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.14204E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36414E+20 3.8E-06  2.36414E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37701E+19 1.2E-07  9.37701E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33789E+20 0.00030  1.19582E+20 0.00020  1.42064E+19 0.00201 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27559E+20 0.00017  2.13352E+20 0.00011  1.42064E+19 0.00201 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49924E+20 0.00037  2.49924E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.31209E+22 0.00025  3.13960E+22 0.00023  1.72491E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.26465E+19 0.00156 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.50205E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12504E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50754E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50754E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.96013E-01 0.05556 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.42193E-01 0.01268 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.54904E-04 0.02672 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.41231E+03 0.02766 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09393E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.96697E-01 0.04059 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.33675E-01 0.04059 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52121E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.45729E-01 0.00048  1.04764E-01 0.00047  3.41413E-04 0.00867 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.45982E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.45981E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.45982E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04024E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01239E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01357E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35134E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34958E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56666E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.56350E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.46287E-03 0.00561  2.74523E-04 0.02049  8.48578E-04 0.01214  6.39167E-04 0.01204  1.35401E-03 0.00883  2.89288E-04 0.01796  5.72979E-05 0.04704 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.56979E-01 0.01619  1.24764E-02 2.6E-05  3.23238E-02 9.8E-05  1.06151E-01 0.00061  2.97793E-01 0.00028  1.23706E+00 0.00030  5.36409E+00 0.03780 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21589E-03 0.00782  2.48733E-04 0.02840  7.87667E-04 0.01658  5.85746E-04 0.01787  1.26589E-03 0.01251  2.73656E-04 0.02786  5.42045E-05 0.06586 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.58460E-01 0.02219  1.24762E-02 3.8E-05  3.23219E-02 0.00014  1.06009E-01 0.00077  2.97844E-01 0.00040  1.23679E+00 0.00039  6.37879E+00 0.02937 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.58454E-07 0.00131  8.58164E-07 0.00131  9.46274E-07 0.01722 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.11801E-07 0.00119  8.11528E-07 0.00120  8.94652E-07 0.01716 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24932E-03 0.00895  2.56135E-04 0.03218  7.82960E-04 0.01865  6.10905E-04 0.01953  1.27307E-03 0.01374  2.71959E-04 0.03197  5.42882E-05 0.07383 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.62645E-01 0.02611  1.24763E-02 4.0E-05  3.23234E-02 0.00017  1.06231E-01 0.00116  2.97759E-01 0.00047  1.23761E+00 0.00058  6.90987E+00 0.03782 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.62981E-07 0.01968  7.62853E-07 0.01968  7.80308E-07 0.05354 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21823E-07 0.01967  7.21701E-07 0.01967  7.38444E-07 0.05362 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85818E-03 0.04296  1.84305E-04 0.14742  7.53249E-04 0.08014  4.98657E-04 0.09388  1.18155E-03 0.06313  1.96447E-04 0.12215  4.39661E-05 0.38110 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.43669E-01 0.09987  1.24753E-02 0.00016  3.23391E-02 0.00062  1.05954E-01 0.00321  2.98014E-01 0.00179  1.23499E+00 0.00179  6.71210E+00 0.11476 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.84405E-03 0.04262  1.83368E-04 0.14230  7.48312E-04 0.08020  5.03582E-04 0.09314  1.16273E-03 0.06249  2.01440E-04 0.12132  4.46169E-05 0.35261 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.46481E-01 0.10031  1.24754E-02 0.00016  3.23406E-02 0.00061  1.05945E-01 0.00320  2.97962E-01 0.00177  1.23511E+00 0.00178  6.69985E+00 0.11441 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77770E+03 0.03890 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.54897E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08452E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20405E-03 0.00608 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.74838E+03 0.00611 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77299E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.85310E-05 0.00883  6.85269E-05 0.00884  6.29085E-07 0.60139 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.07007E-05 0.03185  6.07720E-05 0.03180  1.79305E-07 0.82871 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52775E-04 0.02477  1.53006E-04 0.02480  8.81152E-05 0.57656 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68540E+01 0.01127 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49664E+01 0.00028  4.04269E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 22:38:52 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02103E+00  1.00669E+00  9.96126E-01  9.99934E-01  1.00345E+00  9.92598E-01  9.97763E-01  9.95714E-01  9.97557E-01  9.93877E-01  9.96405E-01  9.95891E-01  1.00273E+00  9.95215E-01  1.00469E+00  1.00518E+00  9.94381E-01  1.00078E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76065E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72393E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40479E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54147E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58902E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51062E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50149E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20935E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.86268E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333101 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33377E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33377E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.86085E+03 ;
RUNNING_TIME              (idx, 1)        =  3.12100E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.21917E-01  1.29850E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.10028E+02  4.98199E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.08650E-01  1.73667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.68500E-02  1.32833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.71276E+01  6.66090E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.05479E+02  6.96187E+02 ];
CPU_USAGE                 (idx, 1)        = 9.16646 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99056E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.09128E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.62566E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67226E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.90785E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97818E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37565E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61519E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50241E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26629E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.92582E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.67727E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.58218E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.94057E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06672E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07122E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.31132E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30624E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50506E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.86645E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78847E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.12364E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55309E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.79497E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.86235E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.61067E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43282E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.78949E-03 -1.27830E+27  2.22075E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.84795E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.62618E+18 0.00398  1.73482E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  9.03506E+19 0.00051  9.63867E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  8.00177E+17 0.00565  8.53603E-03 0.00561 ];
PU239_FISS                (idx, [1:   4]) = [  2.54021E+13 1.00000  2.72287E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00115E+20 0.00049  7.30384E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31945E+19 0.00138  9.62626E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  2.88880E+17 0.00915  2.10751E-03 0.00915 ];
PU239_CAPT                (idx, [1:   4]) = [  2.51788E+13 1.00000  1.82488E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.61741E+17 0.01270  1.17999E-03 0.01270 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001295 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15274E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001295 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5397433 5.40314E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3691299 3.69500E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912563 9.13385E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001295 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.08033E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36381E+20 3.7E-06  2.36381E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37629E+19 1.1E-07  9.37629E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36952E+20 0.00028  1.22645E+20 0.00020  1.43064E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30715E+20 0.00017  2.16408E+20 0.00011  1.43064E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53689E+20 0.00034  2.53689E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36833E+22 0.00024  3.19522E+22 0.00021  1.73115E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.31721E+19 0.00138 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53887E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14323E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49950E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49950E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.15246E-01 0.05865 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27135E-01 0.01442 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.46668E-04 0.02708 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.17022E+03 0.02791 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08664E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.39144E-01 0.04496 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.80779E-01 0.04496 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52105E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.31374E-01 0.00043  1.03162E-01 0.00041  3.40162E-04 0.00967 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.32130E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.31807E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.32130E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02587E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00818E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00703E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35705E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35841E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.60928E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.60830E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.56250E-03 0.00596  2.77055E-04 0.01946  8.69551E-04 0.01196  6.57117E-04 0.01317  1.40361E-03 0.00867  3.00764E-04 0.01833  5.44055E-05 0.04539 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.49824E-01 0.01422  1.24337E-02 0.00334  3.23183E-02 0.00011  1.06221E-01 0.00061  2.98039E-01 0.00026  1.23904E+00 0.00040  5.28001E+00 0.03838 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28794E-03 0.00835  2.54165E-04 0.03034  7.92146E-04 0.01665  6.02767E-04 0.01787  1.30475E-03 0.01337  2.83305E-04 0.02721  5.08065E-05 0.06135 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56753E-01 0.02053  1.24762E-02 3.2E-05  3.23158E-02 0.00015  1.06107E-01 0.00080  2.98123E-01 0.00039  1.23904E+00 0.00056  6.56852E+00 0.02844 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.74491E-07 0.00114  8.74175E-07 0.00114  9.65333E-07 0.01773 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14440E-07 0.00108  8.14145E-07 0.00108  8.98960E-07 0.01773 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28556E-03 0.00966  2.44916E-04 0.03358  8.20315E-04 0.01885  6.02787E-04 0.02130  1.29065E-03 0.01474  2.75369E-04 0.03097  5.15285E-05 0.07024 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.52130E-01 0.02329  1.24759E-02 4.8E-05  3.23200E-02 0.00018  1.06210E-01 0.00104  2.97961E-01 0.00043  1.23958E+00 0.00076  6.49305E+00 0.03995 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.78948E-07 0.01970  7.78821E-07 0.01970  8.25158E-07 0.05324 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.25573E-07 0.01971  7.25455E-07 0.01971  7.68585E-07 0.05323 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.74811E-03 0.04668  2.20756E-04 0.14288  7.10816E-04 0.08226  4.62154E-04 0.09974  1.07512E-03 0.06455  2.45069E-04 0.13717  3.41948E-05 0.33643 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.25979E-01 0.09002  1.24773E-02 0.00012  3.23429E-02 0.00068  1.06344E-01 0.00356  2.98155E-01 0.00179  1.24088E+00 0.00148  6.55826E+00 0.13009 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.74840E-03 0.04645  2.21755E-04 0.13937  7.07878E-04 0.08074  4.63343E-04 0.10074  1.07385E-03 0.06508  2.48946E-04 0.13051  3.26315E-05 0.33837 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.29047E-01 0.08623  1.24774E-02 0.00012  3.23436E-02 0.00068  1.06346E-01 0.00353  2.98140E-01 0.00175  1.24085E+00 0.00148  6.55826E+00 0.13009 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.54211E+03 0.04279 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.68928E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09253E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17781E-03 0.00652 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.65737E+03 0.00651 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78769E-09 0.00068 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74221E-05 0.00983  6.73322E-05 0.00991  5.69654E-07 0.71069 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.04784E-05 0.02802  6.05288E-05 0.02800  1.36169E-07 0.71564 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.46657E-04 0.02562  1.46980E-04 0.02571  5.66459E-05 0.70799 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71410E+01 0.01081 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50149E+01 0.00025  4.04246E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 23:29:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01544E+00  1.00337E+00  1.00143E+00  9.97007E-01  1.00084E+00  9.97237E-01  1.00426E+00  9.96493E-01  9.99540E-01  9.92299E-01  9.85787E-01  9.97326E-01  1.00358E+00  9.99790E-01  1.00650E+00  1.00421E+00  9.94357E-01  1.00053E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76414E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72359E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40073E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53764E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58881E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51689E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50777E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23461E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.90187E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333447 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33361E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33361E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.30390E+03 ;
RUNNING_TIME              (idx, 1)        =  3.62941E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.05557E+00  1.33650E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.60695E+02  5.06672E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.26883E-01  1.82333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.68500E-02  1.32833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.42522E+01  7.12448E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.55856E+02  7.05659E+02 ];
CPU_USAGE                 (idx, 1)        = 9.10314 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99083E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.02062E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.70357E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68492E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.64901E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00789E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39088E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.66269E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51292E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.74538E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22624E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.06085E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.83698E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23180E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33936E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32968E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30664E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29945E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49740E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.59536E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27999E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.16016E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.65462E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.82384E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94842E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.72958E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.17974E-02 -2.60483E+27  2.23401E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01896E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.63164E+18 0.00366  1.74199E-02 0.00359 ];
U233_FISS                 (idx, [1:   4]) = [  8.80697E+19 0.00058  9.40283E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.26073E+18 0.00335  2.41381E-02 0.00335 ];
PU239_FISS                (idx, [1:   4]) = [  6.69610E+14 0.19543  7.13437E-06 0.19558 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00925E+20 0.00054  7.16996E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28653E+19 0.00134  9.14013E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  8.03577E+17 0.00585  5.70915E-03 0.00587 ];
U238_CAPT                 (idx, [1:   4]) = [  2.55752E+13 1.00000  1.80034E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  2.84094E+14 0.29646  2.02713E-06 0.29644 ];
PU240_CAPT                (idx, [1:   4]) = [  5.13560E+13 0.70595  3.68473E-07 0.70599 ];
SM149_CAPT                (idx, [1:   4]) = [  1.68718E+17 0.01244  1.19859E-03 0.01243 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000835 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14106E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000835 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5457055 5.46320E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3631685 3.63525E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912095 9.12956E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000835 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.19792E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36286E+20 3.9E-06  2.36286E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37414E+19 1.1E-07  9.37414E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40695E+20 0.00028  1.26129E+20 0.00021  1.45660E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34437E+20 0.00017  2.19871E+20 0.00012  1.45660E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57653E+20 0.00035  2.57653E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.42753E+22 0.00024  3.25157E+22 0.00022  1.75961E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35233E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.57960E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16271E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47905E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47905E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.11698E-01 0.05221 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33471E-01 0.01440 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.55599E-04 0.02893 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.29949E+03 0.02715 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08707E-01 0.00012 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.78980E-01 0.04029 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.16925E-01 0.04029 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52061E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99747E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.16187E-01 0.00048  1.01471E-01 0.00047  3.38552E-04 0.00967 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.17030E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.17103E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.17030E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00916E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.97991E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.98051E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.39597E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.39493E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.70491E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.70329E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.68699E-03 0.00556  2.86712E-04 0.01979  8.97949E-04 0.01070  6.69361E-04 0.01249  1.46283E-03 0.00862  3.05940E-04 0.01950  6.42009E-05 0.04204 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.58847E-01 0.01446  1.24756E-02 2.8E-05  3.22997E-02 0.00011  1.06463E-01 0.00059  2.98818E-01 0.00030  1.24513E+00 0.00059  5.38297E+00 0.03358 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.33567E-03 0.00825  2.59239E-04 0.02689  8.08660E-04 0.01636  5.95388E-04 0.01869  1.33297E-03 0.01284  2.79053E-04 0.02666  6.03668E-05 0.05965 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.64263E-01 0.02056  1.24757E-02 3.6E-05  3.23043E-02 0.00015  1.06343E-01 0.00077  2.98625E-01 0.00038  1.24532E+00 0.00077  6.26982E+00 0.02668 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82484E-07 0.00154  8.82016E-07 0.00154  1.01698E-06 0.01932 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.08455E-07 0.00143  8.08025E-07 0.00143  9.31879E-07 0.01939 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.31424E-03 0.00974  2.55960E-04 0.03371  7.81127E-04 0.01881  6.16945E-04 0.02253  1.32726E-03 0.01495  2.78453E-04 0.03336  5.44966E-05 0.07122 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.48681E-01 0.02328  1.24751E-02 5.8E-05  3.22950E-02 0.00019  1.06547E-01 0.00102  2.98747E-01 0.00049  1.24431E+00 0.00098  6.18169E+00 0.04018 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.82853E-07 0.01961  7.82628E-07 0.01961  8.67707E-07 0.05975 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.17294E-07 0.01961  7.17087E-07 0.01961  7.95196E-07 0.05978 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00296E-03 0.04356  2.46247E-04 0.13515  6.29003E-04 0.08581  5.85120E-04 0.08844  1.22735E-03 0.06216  2.50816E-04 0.12932  6.44335E-05 0.26824 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.70442E-01 0.10595  1.24743E-02 0.00018  3.23038E-02 0.00085  1.06423E-01 0.00330  2.99158E-01 0.00183  1.24511E+00 0.00291  6.34879E+00 0.11208 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99075E-03 0.04288  2.49870E-04 0.13421  6.43180E-04 0.08303  5.91314E-04 0.08677  1.19578E-03 0.06204  2.44935E-04 0.13079  6.56757E-05 0.26338 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.68988E-01 0.10346  1.24743E-02 0.00018  3.23044E-02 0.00085  1.06398E-01 0.00329  2.99129E-01 0.00181  1.24470E+00 0.00288  6.34026E+00 0.11197 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.85418E+03 0.03901 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.76569E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.03047E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34660E-03 0.00606 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81782E+03 0.00602 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.80047E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.62227E-05 0.01076  6.62107E-05 0.01075  9.22750E-07 0.53397 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.71800E-05 0.03012  5.71681E-05 0.03014  7.61884E-07 0.53603 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.53992E-04 0.02726  1.54157E-04 0.02718  1.09716E-04 0.49794 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70571E+01 0.01090 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50777E+01 0.00025  4.02282E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 00:21:10 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01235E+00  9.98249E-01  9.96980E-01  9.90139E-01  1.00096E+00  1.00143E+00  1.00151E+00  1.00361E+00  9.99842E-01  9.92800E-01  9.95995E-01  9.97436E-01  9.95152E-01  1.00474E+00  1.00579E+00  1.00093E+00  9.99553E-01  1.00253E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.77428E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72257E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40115E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53847E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58655E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50053E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49142E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20174E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.91653E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333983 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33393E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33393E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.75247E+03 ;
RUNNING_TIME              (idx, 1)        =  4.14409E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.19132E+00  1.35750E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.11984E+02  5.12894E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.45050E-01  1.81667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.14551E+01  7.20283E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.07244E+02  7.12171E+02 ];
CPU_USAGE                 (idx, 1)        = 9.05499 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99103E+00 1.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.96693E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.86881E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.72892E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.47352E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01810E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39213E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.81694E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55608E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.08621E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77211E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.00565E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.19688E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04537E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84549E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38191E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37708E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34756E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56646E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.87661E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30576E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.25948E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.90647E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.40070E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09729E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.78446E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57630E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.24071E-02 -4.94741E+27  2.25744E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02878E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.59874E+18 0.00418  1.70562E-02 0.00412 ];
U233_FISS                 (idx, [1:   4]) = [  8.47667E+19 0.00052  9.04377E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.64983E+18 0.00222  4.96096E-02 0.00218 ];
PU239_FISS                (idx, [1:   4]) = [  1.86943E+16 0.03856  1.99379E-04 0.03852 ];
PU240_FISS                (idx, [1:   4]) = [  7.51599E+14 0.18955  8.01279E-06 0.18960 ];
PU241_FISS                (idx, [1:   4]) = [  4.15508E+14 0.24366  4.42212E-06 0.24367 ];
TH232_CAPT                (idx, [1:   4]) = [  9.88850E+19 0.00054  6.94572E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23660E+19 0.00144  8.68589E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.64269E+18 0.00391  1.15382E-02 0.00388 ];
U238_CAPT                 (idx, [1:   4]) = [  2.86246E+14 0.32361  2.00038E-06 0.32305 ];
PU239_CAPT                (idx, [1:   4]) = [  9.16646E+15 0.05059  6.43893E-05 0.05062 ];
PU240_CAPT                (idx, [1:   4]) = [  1.71575E+15 0.12847  1.20416E-05 0.12849 ];
PU241_CAPT                (idx, [1:   4]) = [  1.04239E+14 0.49751  7.29768E-07 0.49750 ];
SM149_CAPT                (idx, [1:   4]) = [  1.69949E+17 0.01299  1.19366E-03 0.01295 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001800 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11707E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001800 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5481297 5.48667E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3608945 3.61220E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911558 9.12300E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001800 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -8.56817E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36161E+20 4.4E-06  2.36161E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37035E+19 1.2E-07  9.37035E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42352E+20 0.00030  1.27686E+20 0.00020  1.46656E+19 0.00196 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36055E+20 0.00018  2.21390E+20 0.00012  1.46656E+19 0.00196 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59482E+20 0.00035  2.59482E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44350E+22 0.00024  3.26637E+22 0.00022  1.77134E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36735E+19 0.00155 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59729E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16669E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42526E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42526E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.98735E-01 0.05256 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.43483E-01 0.01238 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.53946E-04 0.02601 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.25420E+03 0.02592 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08771E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99999E-01 3.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.91360E-01 0.03877 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.28319E-01 0.03877 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52030E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99827E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.10393E-01 0.00042  1.00810E-01 0.00042  3.42258E-04 0.00931 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.10298E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.10159E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.10298E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00170E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92243E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92208E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47859E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47886E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.84275E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.84263E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.82727E-03 0.00528  2.83772E-04 0.02058  9.14548E-04 0.01102  6.93887E-04 0.01338  1.52558E-03 0.00901  3.35784E-04 0.01741  7.37048E-05 0.03904 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.86487E-01 0.01472  1.24342E-02 0.00334  3.22736E-02 0.00012  1.06483E-01 0.00056  2.99963E-01 0.00029  1.25361E+00 0.00071  5.82162E+00 0.03055 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.40130E-03 0.00756  2.58842E-04 0.02686  8.04364E-04 0.01567  6.04920E-04 0.02026  1.36965E-03 0.01260  2.95119E-04 0.02550  6.84061E-05 0.06002 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.87656E-01 0.02226  1.24763E-02 3.9E-05  3.22694E-02 0.00016  1.06567E-01 0.00079  2.99925E-01 0.00042  1.25249E+00 0.00094  6.49297E+00 0.02439 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64117E-07 0.00132  8.63785E-07 0.00132  9.63531E-07 0.01793 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.86636E-07 0.00122  7.86334E-07 0.00123  8.77080E-07 0.01790 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.37559E-03 0.00955  2.53795E-04 0.03582  7.86630E-04 0.01971  6.11965E-04 0.02088  1.35182E-03 0.01489  2.97616E-04 0.03010  7.37565E-05 0.06356 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.05884E-01 0.02566  1.24759E-02 5.0E-05  3.22620E-02 0.00022  1.06502E-01 0.00098  3.00212E-01 0.00048  1.25406E+00 0.00120  6.69847E+00 0.03153 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.59899E-07 0.01966  7.59510E-07 0.01966  8.58649E-07 0.05645 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.92070E-07 0.01965  6.91715E-07 0.01965  7.82306E-07 0.05642 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92012E-03 0.04721  2.37300E-04 0.13763  7.26830E-04 0.08565  5.05902E-04 0.10145  1.12485E-03 0.06421  2.88452E-04 0.12170  3.67863E-05 0.32783 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.44566E-01 0.07739  1.24761E-02 0.00015  3.22613E-02 0.00093  1.06080E-01 0.00287  2.99444E-01 0.00178  1.26179E+00 0.00379  6.38387E+00 0.11930 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92128E-03 0.04647  2.39765E-04 0.14078  7.28154E-04 0.08141  5.09218E-04 0.09879  1.12293E-03 0.06231  2.86597E-04 0.12003  3.46163E-05 0.31440 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.42318E-01 0.07053  1.24760E-02 0.00016  3.22637E-02 0.00092  1.06069E-01 0.00285  2.99423E-01 0.00176  1.26185E+00 0.00379  6.38383E+00 0.11930 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.86207E+03 0.04299 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.57833E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.80920E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39060E-03 0.00687 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95314E+03 0.00691 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77497E-09 0.00070 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71951E-05 0.00997  6.71544E-05 0.00996  1.96755E-06 0.35757 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.82714E-05 0.02978  5.82546E-05 0.02989  2.00764E-06 0.46302 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52077E-04 0.02573  1.51859E-04 0.02585  2.11091E-04 0.35031 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63436E+01 0.01099 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49142E+01 0.00029  3.97114E+01 0.00037 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 01:12:29 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01835E+00  9.99059E-01  9.95786E-01  9.97030E-01  1.00763E+00  9.98535E-01  1.00328E+00  9.98221E-01  9.98579E-01  9.93493E-01  9.93782E-01  9.96002E-01  1.00495E+00  9.97515E-01  9.99412E-01  9.97202E-01  1.00202E+00  9.99152E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.77728E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72227E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40165E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53906E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58810E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49270E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48364E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18541E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.91438E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333857 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33367E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33367E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.19968E+03 ;
RUNNING_TIME              (idx, 1)        =  4.65718E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.32865E+00  1.37333E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.63111E+02  5.11269E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.63317E-01  1.82667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.86288E+01  7.17372E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.58584E+02  7.15809E+02 ];
CPU_USAGE                 (idx, 1)        = 9.01765 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99098E+00 1.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.92531E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 182 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.90321E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73786E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.40804E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01466E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38909E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85450E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56505E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.00775E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98846E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.83276E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.38576E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13478E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04196E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81843E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38505E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35224E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57359E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92954E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73586E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27679E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96751E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.82659E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.12224E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.78214E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21958E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.75140E-02 -6.07499E+27  2.26872E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02677E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.58014E+18 0.00393  1.68586E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  8.37118E+19 0.00053  8.93100E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  5.32909E+18 0.00224  5.68537E-02 0.00216 ];
PU239_FISS                (idx, [1:   4]) = [  4.44738E+16 0.02426  4.74423E-04 0.02422 ];
PU240_FISS                (idx, [1:   4]) = [  1.42563E+15 0.12746  1.52202E-05 0.12742 ];
PU241_FISS                (idx, [1:   4]) = [  2.39035E+15 0.10233  2.54881E-05 0.10234 ];
TH232_CAPT                (idx, [1:   4]) = [  9.77176E+19 0.00057  6.86012E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22193E+19 0.00135  8.57872E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.88269E+18 0.00368  1.32175E-02 0.00367 ];
U238_CAPT                 (idx, [1:   4]) = [  2.35191E+14 0.36505  1.64715E-06 0.36591 ];
PU239_CAPT                (idx, [1:   4]) = [  2.00777E+16 0.03777  1.40944E-04 0.03777 ];
PU240_CAPT                (idx, [1:   4]) = [  4.84522E+15 0.07418  3.40668E-05 0.07429 ];
PU241_CAPT                (idx, [1:   4]) = [  4.67493E+14 0.24251  3.27622E-06 0.24235 ];
SM149_CAPT                (idx, [1:   4]) = [  1.73201E+17 0.01261  1.21614E-03 0.01265 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001015 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11965E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001015 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5485367 5.49113E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3609769 3.61337E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905879 9.06699E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001015 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.83122E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36147E+20 4.1E-06  2.36147E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36893E+19 1.1E-07  9.36893E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42351E+20 0.00028  1.27660E+20 0.00020  1.46908E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36040E+20 0.00017  2.21349E+20 0.00011  1.46908E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59405E+20 0.00033  2.59405E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43841E+22 0.00024  3.26083E+22 0.00022  1.77579E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35207E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59561E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16432E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39640E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39640E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.28872E-01 0.05387 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24054E-01 0.01461 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.51822E-04 0.02687 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.94828E+03 0.02784 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09332E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.67943E-01 0.04090 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.07348E-01 0.04090 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52054E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99858E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.10732E-01 0.00044  1.00852E-01 0.00043  3.46613E-04 0.00962 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.10817E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.10374E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.10817E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00169E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90045E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90155E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.51142E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50954E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.89380E-01 0.00138 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.88966E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.83909E-03 0.00543  2.93730E-04 0.01923  9.14512E-04 0.01065  6.90023E-04 0.01278  1.52720E-03 0.00867  3.41252E-04 0.01597  7.23728E-05 0.04153 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.89605E-01 0.01508  1.24766E-02 3.8E-05  3.22594E-02 0.00012  1.06560E-01 0.00053  3.00372E-01 0.00030  1.25538E+00 0.00072  6.13645E+00 0.03019 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.38207E-03 0.00769  2.59524E-04 0.02645  8.02924E-04 0.01480  6.01538E-04 0.01835  1.36063E-03 0.01226  2.93329E-04 0.02413  6.41272E-05 0.05889 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.82871E-01 0.02118  1.24764E-02 4.8E-05  3.22592E-02 0.00017  1.06558E-01 0.00076  3.00440E-01 0.00043  1.25663E+00 0.00098  6.91703E+00 0.02292 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.56243E-07 0.00130  8.55875E-07 0.00131  9.62305E-07 0.02039 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.79767E-07 0.00124  7.79431E-07 0.00124  8.76647E-07 0.02056 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41719E-03 0.00961  2.65452E-04 0.03101  8.21921E-04 0.01835  6.07561E-04 0.02182  1.35951E-03 0.01455  2.95219E-04 0.02809  6.75308E-05 0.06880 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.93594E-01 0.02509  1.24762E-02 8.3E-05  3.22574E-02 0.00021  1.06543E-01 0.00100  3.00326E-01 0.00052  1.25546E+00 0.00131  7.07313E+00 0.03164 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.52227E-07 0.01959  7.51831E-07 0.01959  9.08539E-07 0.06016 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.85295E-07 0.01959  6.84935E-07 0.01959  8.27254E-07 0.05994 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.04700E-03 0.04498  1.97189E-04 0.14281  7.89580E-04 0.07775  6.56998E-04 0.08752  1.05805E-03 0.07118  2.80200E-04 0.11810  6.49893E-05 0.27788 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.20291E-01 0.09958  1.24778E-02 0.00011  3.22746E-02 0.00080  1.06732E-01 0.00319  3.00678E-01 0.00212  1.25816E+00 0.00370  6.85524E+00 0.09561 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.05089E-03 0.04437  1.98014E-04 0.14123  7.91726E-04 0.07760  6.51555E-04 0.08717  1.06892E-03 0.06903  2.76463E-04 0.11647  6.42145E-05 0.26547 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.14599E-01 0.09609  1.24778E-02 0.00010  3.22732E-02 0.00081  1.06754E-01 0.00320  3.00612E-01 0.00209  1.25865E+00 0.00370  6.85524E+00 0.09561 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.07600E+03 0.04139 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.48580E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.72786E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.48878E-03 0.00549 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.11234E+03 0.00559 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76265E-09 0.00071 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79249E-05 0.00853  6.79265E-05 0.00852  9.42685E-07 0.53416 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.72623E-05 0.03136  5.72289E-05 0.03137  8.78958E-07 0.56726 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.51860E-04 0.02607  1.52046E-04 0.02608  1.00425E-04 0.49822 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65646E+01 0.01207 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48364E+01 0.00026  3.95244E+01 0.00038 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 02:04:56 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00905E+00  9.98241E-01  1.00148E+00  9.94072E-01  9.91544E-01  1.00088E+00  1.00070E+00  1.00198E+00  1.00093E+00  9.96791E-01  9.92886E-01  1.00069E+00  9.98521E-01  1.00259E+00  1.00549E+00  9.98594E-01  1.00456E+00  1.00099E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.78129E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72187E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40140E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53897E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58811E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48624E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47716E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17390E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.92500E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333742 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33397E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33397E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.65663E+03 ;
RUNNING_TIME              (idx, 1)        =  5.18166E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.46780E+00  1.39150E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.15369E+02  5.22582E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.82683E-01  1.93667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.59781E+01  7.34920E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.10857E+02  7.15999E+02 ];
CPU_USAGE                 (idx, 1)        = 8.98676 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99099E+00 1.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.89093E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.90266E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73841E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.96335E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99954E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37848E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86905E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56666E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.06549E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.16877E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.81042E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.54926E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21167E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20529E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20333E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37655E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34456E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56375E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95361E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11796E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26867E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99126E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.15826E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10084E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.78110E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.26774E-02 -7.21506E+27  2.28012E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02676E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.55879E+18 0.00425  1.66516E-02 0.00421 ];
U233_FISS                 (idx, [1:   4]) = [  8.27799E+19 0.00052  8.84305E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.84829E+18 0.00215  6.24745E-02 0.00208 ];
U238_FISS                 (idx, [1:   4]) = [  5.17439E+13 0.70601  5.56944E-07 0.70616 ];
PU239_FISS                (idx, [1:   4]) = [  8.21533E+16 0.01801  8.77879E-04 0.01805 ];
PU240_FISS                (idx, [1:   4]) = [  3.40027E+15 0.08803  3.63429E-05 0.08805 ];
PU241_FISS                (idx, [1:   4]) = [  3.57778E+15 0.08668  3.82115E-05 0.08663 ];
TH232_CAPT                (idx, [1:   4]) = [  9.68324E+19 0.00052  6.79715E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20730E+19 0.00130  8.47465E-02 0.00124 ];
U235_CAPT                 (idx, [1:   4]) = [  2.05300E+18 0.00373  1.44107E-02 0.00368 ];
U238_CAPT                 (idx, [1:   4]) = [  5.98328E+14 0.22788  4.21244E-06 0.22741 ];
PU239_CAPT                (idx, [1:   4]) = [  3.83771E+16 0.02548  2.69328E-04 0.02540 ];
PU240_CAPT                (idx, [1:   4]) = [  9.57687E+15 0.05091  6.72339E-05 0.05094 ];
PU241_CAPT                (idx, [1:   4]) = [  1.11552E+15 0.15601  7.83288E-06 0.15589 ];
SM149_CAPT                (idx, [1:   4]) = [  1.73326E+17 0.01182  1.21667E-03 0.01182 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001903 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10189E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001903 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487329 5.49258E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3606062 3.60918E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908512 9.09255E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001903 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.45058E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36155E+20 4.3E-06  2.36155E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36768E+19 1.1E-07  9.36768E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42381E+20 0.00030  1.27697E+20 0.00019  1.46834E+19 0.00196 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36057E+20 0.00018  2.21374E+20 0.00011  1.46834E+19 0.00196 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59370E+20 0.00035  2.59370E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43548E+22 0.00026  3.25804E+22 0.00022  1.77444E+21 0.00192 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35840E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59641E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16246E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.36823E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.36823E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.52904E-01 0.05623 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24281E-01 0.01388 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.57747E-04 0.02570 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.98104E+03 0.02890 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09077E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.30420E-01 0.04432 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.73127E-01 0.04432 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52095E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99884E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09695E-01 0.00045  1.00748E-01 0.00044  3.49303E-04 0.00822 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.10558E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.10528E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.10558E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00164E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88355E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88388E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53724E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53643E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.92890E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.92671E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85782E-03 0.00548  2.87012E-04 0.01929  9.17712E-04 0.01115  6.94783E-04 0.01196  1.53188E-03 0.00863  3.44868E-04 0.01888  8.15611E-05 0.03792 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05840E-01 0.01522  1.24769E-02 5.8E-05  3.22469E-02 0.00013  1.06698E-01 0.00059  3.00442E-01 0.00031  1.25843E+00 0.00070  6.32630E+00 0.02661 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44683E-03 0.00819  2.58741E-04 0.02918  8.22666E-04 0.01748  6.04746E-04 0.01737  1.38288E-03 0.01286  3.04384E-04 0.02568  7.34110E-05 0.05728 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.08867E-01 0.02391  1.24771E-02 7.8E-05  3.22398E-02 0.00017  1.06627E-01 0.00076  3.00436E-01 0.00040  1.25787E+00 0.00103  6.90153E+00 0.02204 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.48494E-07 0.00126  8.47994E-07 0.00126  9.90404E-07 0.01625 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.71821E-07 0.00116  7.71366E-07 0.00116  9.00925E-07 0.01621 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45491E-03 0.00833  2.67028E-04 0.03257  8.35942E-04 0.01854  6.26150E-04 0.02123  1.35165E-03 0.01383  2.99091E-04 0.02925  7.50479E-05 0.06455 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07772E-01 0.02587  1.24775E-02 7.7E-05  3.22477E-02 0.00020  1.06664E-01 0.00092  3.00449E-01 0.00052  1.25945E+00 0.00124  7.05568E+00 0.02997 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.50641E-07 0.01959  7.50281E-07 0.01960  8.16346E-07 0.05859 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.83074E-07 0.01958  6.82746E-07 0.01959  7.42913E-07 0.05861 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98273E-03 0.04408  2.55911E-04 0.14409  7.22249E-04 0.07858  4.95864E-04 0.09957  1.10281E-03 0.06232  3.42546E-04 0.10749  6.33488E-05 0.26388 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.26652E-01 0.08615  1.24733E-02 0.00021  3.22400E-02 0.00077  1.07109E-01 0.00348  2.99498E-01 0.00190  1.25627E+00 0.00333  7.07785E+00 0.09093 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94771E-03 0.04334  2.43475E-04 0.14611  7.15266E-04 0.07867  4.82603E-04 0.09569  1.11215E-03 0.06164  3.29473E-04 0.10581  6.47522E-05 0.25365 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.30425E-01 0.08616  1.24732E-02 0.00021  3.22386E-02 0.00076  1.07142E-01 0.00349  2.99529E-01 0.00190  1.25656E+00 0.00335  7.07785E+00 0.09093 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.99355E+03 0.03976 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.43156E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.66969E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40602E-03 0.00623 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.04003E+03 0.00623 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76171E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75652E-05 0.00815  6.75482E-05 0.00819  1.03434E-06 0.45192 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.89962E-05 0.02736  5.88477E-05 0.02777  2.23014E-06 0.52354 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.58795E-04 0.02448  1.58913E-04 0.02464  1.26106E-04 0.44533 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64801E+01 0.01131 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47716E+01 0.00027  3.93570E+01 0.00039 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 02:58:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01198E+00  1.00651E+00  9.96554E-01  9.96372E-01  9.94021E-01  1.00015E+00  1.00087E+00  1.00457E+00  1.00078E+00  1.00439E+00  9.95657E-01  9.97583E-01  9.96123E-01  9.95006E-01  1.00452E+00  9.96500E-01  9.98023E-01  1.00039E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.79678E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72032E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39624E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53452E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59064E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47291E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46380E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16540E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.99383E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333487 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33393E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33393E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.12116E+03 ;
RUNNING_TIME              (idx, 1)        =  5.71417E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.61172E+00  1.43917E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.68425E+02  5.30558E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.01650E-01  1.89667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.33741E+01  7.39598E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.64062E+02  7.21325E+02 ];
CPU_USAGE                 (idx, 1)        = 8.96221 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99115E+00 1.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.86361E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.92468E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74459E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.26330E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96489E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35773E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92575E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57497E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.57726E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.74840E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.08618E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03596E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44228E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70281E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34308E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37136E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33772E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55607E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99278E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27134E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26507E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04870E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.40590E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07784E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.80324E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43599E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.32093E-02 -1.17484E+28  2.32545E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02438E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.54860E+18 0.00399  1.65453E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  8.07145E+19 0.00054  8.62364E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.86251E+18 0.00199  7.33196E-02 0.00192 ];
PU239_FISS                (idx, [1:   4]) = [  2.87585E+17 0.00900  3.07259E-03 0.00897 ];
PU240_FISS                (idx, [1:   4]) = [  2.03838E+16 0.03408  2.17782E-04 0.03407 ];
PU241_FISS                (idx, [1:   4]) = [  3.24536E+16 0.02809  3.46648E-04 0.02804 ];
TH232_CAPT                (idx, [1:   4]) = [  9.46812E+19 0.00051  6.61703E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17687E+19 0.00155  8.22486E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.39554E+18 0.00334  1.67423E-02 0.00334 ];
U238_CAPT                 (idx, [1:   4]) = [  1.01241E+15 0.15805  7.08013E-06 0.15806 ];
PU239_CAPT                (idx, [1:   4]) = [  1.30120E+17 0.01493  9.09267E-04 0.01490 ];
PU240_CAPT                (idx, [1:   4]) = [  5.49413E+16 0.02081  3.84097E-04 0.02087 ];
PU241_CAPT                (idx, [1:   4]) = [  6.63178E+15 0.06492  4.63568E-05 0.06494 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75353E+17 0.01264  1.22535E-03 0.01260 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001781 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09459E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001781 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5495930 5.50111E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3595185 3.59841E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910666 9.11427E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001781 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -8.19564E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36303E+20 4.4E-06  2.36303E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36378E+19 9.7E-08  9.36378E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43089E+20 0.00027  1.28269E+20 0.00020  1.48209E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36727E+20 0.00017  2.21906E+20 0.00011  1.48209E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60108E+20 0.00033  2.60108E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44526E+22 0.00024  3.26678E+22 0.00021  1.78484E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37079E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60435E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16227E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26132E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26132E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.19474E-01 0.05787 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.29781E-01 0.01219 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.76924E-04 0.02440 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.55350E+03 0.03119 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08861E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.26105E-01 0.04464 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.69105E-01 0.04464 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52358E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99968E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07989E-01 0.00045  1.00554E-01 0.00044  3.44838E-04 0.00832 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08345E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08508E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08345E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99420E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84159E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84164E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.60306E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.60268E-02 0.00080 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.02988E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.02582E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.89415E-03 0.00544  2.75661E-04 0.01998  9.05585E-04 0.01113  7.07079E-04 0.01242  1.56743E-03 0.00840  3.54985E-04 0.01750  8.34070E-05 0.03441 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13522E-01 0.01347  1.24776E-02 5.0E-05  3.22157E-02 0.00013  1.06843E-01 0.00056  3.01140E-01 0.00031  1.25797E+00 0.00088  6.64554E+00 0.02341 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46078E-03 0.00732  2.53793E-04 0.02610  8.10333E-04 0.01557  6.19997E-04 0.01747  1.39499E-03 0.01165  3.11750E-04 0.02383  6.99159E-05 0.05075 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.06837E-01 0.01938  1.24777E-02 5.4E-05  3.22109E-02 0.00018  1.06840E-01 0.00079  3.01178E-01 0.00045  1.25934E+00 0.00120  7.15032E+00 0.01990 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.36806E-07 0.00130  8.36339E-07 0.00129  9.79235E-07 0.02301 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.59765E-07 0.00121  7.59341E-07 0.00121  8.88959E-07 0.02297 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42040E-03 0.00841  2.34305E-04 0.03440  7.87855E-04 0.01884  6.29034E-04 0.01975  1.38113E-03 0.01277  3.12854E-04 0.03051  7.52176E-05 0.06036 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.18622E-01 0.02387  1.24769E-02 5.1E-05  3.22225E-02 0.00024  1.06743E-01 0.00101  3.01014E-01 0.00057  1.25953E+00 0.00154  6.94095E+00 0.02958 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.41272E-07 0.01960  7.40925E-07 0.01960  7.96934E-07 0.05491 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.73093E-07 0.01958  6.72778E-07 0.01958  7.23732E-07 0.05490 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01822E-03 0.04340  2.02713E-04 0.14176  6.92833E-04 0.08315  6.08315E-04 0.08465  1.20333E-03 0.06582  2.33109E-04 0.12985  7.79117E-05 0.27220 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.70620E-01 0.12154  1.24784E-02 9.8E-05  3.22948E-02 0.00082  1.06840E-01 0.00312  3.00748E-01 0.00188  1.26021E+00 0.00540  7.03415E+00 0.07895 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.05080E-03 0.04298  2.12005E-04 0.13831  6.85472E-04 0.08374  6.16804E-04 0.08397  1.22898E-03 0.06501  2.31455E-04 0.12692  7.60787E-05 0.25319 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.80693E-01 0.12039  1.24785E-02 9.4E-05  3.22959E-02 0.00082  1.06840E-01 0.00309  3.00737E-01 0.00185  1.26064E+00 0.00537  7.03415E+00 0.07895 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.09118E+03 0.03874 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.30534E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.54068E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44037E-03 0.00585 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.14282E+03 0.00586 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78078E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68543E-05 0.00853  6.68471E-05 0.00854  1.08011E-06 0.44483 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65769E-05 0.02531  5.67272E-05 0.02541  4.69384E-07 0.47438 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.75311E-04 0.02338  1.75492E-04 0.02348  1.35501E-04 0.44436 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59311E+01 0.01124 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46380E+01 0.00027  3.89642E+01 0.00037 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 03:51:37 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01337E+00  1.00028E+00  1.00439E+00  9.95469E-01  9.97664E-01  1.00517E+00  1.00144E+00  9.97482E-01  9.94351E-01  1.00166E+00  9.95910E-01  9.96855E-01  9.91152E-01  9.95072E-01  1.00481E+00  1.00197E+00  9.97556E-01  1.00541E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.83035E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71696E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38269E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52256E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60332E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46531E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45608E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19409E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.02057E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334316 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33402E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33402E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.58655E+03 ;
RUNNING_TIME              (idx, 1)        =  6.24858E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.75527E+00  1.43550E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.21669E+02  5.32438E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.21400E-01  1.97500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.08813E+01  7.50718E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.17389E+02  7.23827E+02 ];
CPU_USAGE                 (idx, 1)        = 8.94052 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99123E+00 1.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.83951E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.91366E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73963E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.09387E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92392E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33734E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95565E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57199E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53592E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39103E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36116E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14236E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69241E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23780E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58100E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34531E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31668E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52798E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02796E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59754E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23431E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02995E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.15261E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01833E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.83535E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06719E-01 -2.35632E+28  2.44360E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01962E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.51974E+18 0.00407  1.62414E-02 0.00403 ];
U233_FISS                 (idx, [1:   4]) = [  7.93809E+19 0.00061  8.48346E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.10529E+18 0.00175  7.59372E-02 0.00173 ];
U238_FISS                 (idx, [1:   4]) = [  7.85227E+13 0.57543  8.40131E-07 0.57543 ];
PU239_FISS                (idx, [1:   4]) = [  5.96656E+17 0.00657  6.37616E-03 0.00654 ];
PU240_FISS                (idx, [1:   4]) = [  7.29654E+16 0.01882  7.79802E-04 0.01881 ];
PU241_FISS                (idx, [1:   4]) = [  1.23243E+17 0.01440  1.31711E-03 0.01438 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28737E+19 0.00052  6.46065E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15801E+19 0.00154  8.05550E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50395E+18 0.00318  1.74195E-02 0.00323 ];
U238_CAPT                 (idx, [1:   4]) = [  2.30360E+15 0.10568  1.60371E-05 0.10578 ];
PU239_CAPT                (idx, [1:   4]) = [  2.86996E+17 0.00982  1.99631E-03 0.00978 ];
PU240_CAPT                (idx, [1:   4]) = [  1.81898E+17 0.01225  1.26525E-03 0.01222 ];
PU241_CAPT                (idx, [1:   4]) = [  2.60410E+16 0.03153  1.81076E-04 0.03147 ];
SM149_CAPT                (idx, [1:   4]) = [  1.79374E+17 0.01162  1.24798E-03 0.01165 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002050 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09421E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002050 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498980 5.50406E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3579563 3.58265E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 923507 9.24230E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002050 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.16650E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36610E+20 4.8E-06  2.36610E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35968E+19 9.9E-08  9.35968E+19 9.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43715E+20 0.00027  1.28486E+20 0.00019  1.52285E+19 0.00171 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37312E+20 0.00017  2.22083E+20 0.00011  1.52285E+19 0.00171 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61178E+20 0.00033  2.61178E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47671E+22 0.00024  3.29524E+22 0.00022  1.81475E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41396E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61451E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16505E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.04370E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.04370E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.74726E-01 0.05070 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23860E-01 0.01141 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.21065E-04 0.02257 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.67859E+03 0.02793 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07581E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.28616E-01 0.03518 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.61252E-01 0.03517 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52797E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00055E+02 9.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05700E-01 0.00044  1.00283E-01 0.00044  3.46676E-04 0.00872 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05975E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05962E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05975E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98239E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81199E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81044E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65121E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.65352E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09651E-01 0.00136 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.09878E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93211E-03 0.00558  2.78854E-04 0.01963  9.16858E-04 0.01057  7.03648E-04 0.01361  1.58194E-03 0.00885  3.68894E-04 0.01749  8.19069E-05 0.03815 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13643E-01 0.01460  1.24783E-02 8.2E-05  3.21989E-02 0.00016  1.06791E-01 0.00054  3.01301E-01 0.00031  1.25899E+00 0.00091  6.42606E+00 0.02550 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43771E-03 0.00799  2.47691E-04 0.02898  7.99846E-04 0.01507  6.27806E-04 0.01849  1.36938E-03 0.01207  3.22827E-04 0.02565  7.01621E-05 0.05186 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09920E-01 0.01992  1.24768E-02 4.9E-05  3.21983E-02 0.00021  1.06667E-01 0.00073  3.01313E-01 0.00041  1.25938E+00 0.00123  6.96563E+00 0.02189 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.30247E-07 0.00127  8.29894E-07 0.00127  9.29588E-07 0.01620 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.51920E-07 0.00122  7.51601E-07 0.00122  8.41887E-07 0.01620 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44362E-03 0.00881  2.46444E-04 0.03440  8.08324E-04 0.01705  6.13954E-04 0.02103  1.38001E-03 0.01428  3.29525E-04 0.02837  6.53626E-05 0.06611 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.01776E-01 0.02394  1.24792E-02 0.00012  3.21953E-02 0.00030  1.06663E-01 0.00097  3.01367E-01 0.00052  1.26031E+00 0.00168  7.02176E+00 0.03110 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.33083E-07 0.01975  7.32797E-07 0.01975  8.08686E-07 0.04963 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.64146E-07 0.01973  6.63889E-07 0.01974  7.32472E-07 0.04958 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.24151E-03 0.04285  3.07650E-04 0.13771  8.39910E-04 0.08156  5.03612E-04 0.08746  1.19103E-03 0.06904  3.34610E-04 0.13398  6.46979E-05 0.24979 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.19379E-01 0.09436  1.24775E-02 0.00012  3.21981E-02 0.00098  1.06640E-01 0.00323  3.01292E-01 0.00201  1.26152E+00 0.00633  7.37129E+00 0.07525 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.24225E-03 0.04231  3.15040E-04 0.13737  8.27858E-04 0.08017  5.04553E-04 0.08660  1.20189E-03 0.06859  3.29847E-04 0.13169  6.30619E-05 0.23257 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.22320E-01 0.09268  1.24775E-02 0.00012  3.22003E-02 0.00099  1.06598E-01 0.00317  3.01220E-01 0.00198  1.26141E+00 0.00633  7.37129E+00 0.07525 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.44961E+03 0.03846 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.22344E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.44755E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.52069E-03 0.00563 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.28167E+03 0.00563 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.87416E-09 0.00084 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.85143E-05 0.00775  6.85490E-05 0.00787  1.29601E-06 0.41615 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65132E-05 0.02256  5.65775E-05 0.02250  8.47522E-07 0.58385 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.19762E-04 0.02116  2.20031E-04 0.02131  1.52403E-04 0.40596 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59529E+01 0.01157 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45608E+01 0.00027  3.86683E+01 0.00037 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 04:45:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01389E+00  1.00522E+00  1.00020E+00  9.96035E-01  9.97250E-01  1.00108E+00  1.00421E+00  9.97250E-01  9.96593E-01  1.00861E+00  9.96255E-01  9.95922E-01  9.99993E-01  9.93541E-01  9.98043E-01  1.00148E+00  9.99474E-01  9.94962E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.86782E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71322E-01 9.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36628E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50793E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62084E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47157E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46225E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.25828E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.04742E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333918 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33381E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33381E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.05358E+03 ;
RUNNING_TIME              (idx, 1)        =  6.78515E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.89968E+00  1.44417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.75123E+02  5.34543E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.39700E-01  1.83000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.84433E+01  7.56190E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.70994E+02  7.24400E+02 ];
CPU_USAGE                 (idx, 1)        = 8.92180 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99122E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.81868E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.91011E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73774E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.35401E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91250E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33161E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96341E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57056E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64076E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56577E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45849E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16069E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76435E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39360E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93081E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33761E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31103E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52006E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04637E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00724E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22388E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01889E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34988E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99952E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.84218E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93016E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61492E-01 -3.56569E+28  2.56453E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01572E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.49486E+18 0.00426  1.59809E-02 0.00421 ];
U233_FISS                 (idx, [1:   4]) = [  7.92036E+19 0.00059  8.46740E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.08510E+18 0.00199  7.57469E-02 0.00197 ];
U238_FISS                 (idx, [1:   4]) = [  2.62794E+14 0.31146  2.79871E-06 0.31148 ];
PU239_FISS                (idx, [1:   4]) = [  6.55184E+17 0.00635  7.00419E-03 0.00631 ];
PU240_FISS                (idx, [1:   4]) = [  8.48543E+16 0.01761  9.07365E-04 0.01766 ];
PU241_FISS                (idx, [1:   4]) = [  1.49677E+17 0.01377  1.60009E-03 0.01375 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22273E+19 0.00055  6.41480E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15432E+19 0.00148  8.02893E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48665E+18 0.00342  1.72953E-02 0.00338 ];
U238_CAPT                 (idx, [1:   4]) = [  2.85166E+15 0.09443  1.98134E-05 0.09447 ];
PU239_CAPT                (idx, [1:   4]) = [  2.95142E+17 0.01008  2.05285E-03 0.01007 ];
PU240_CAPT                (idx, [1:   4]) = [  2.17975E+17 0.01130  1.51610E-03 0.01129 ];
PU241_CAPT                (idx, [1:   4]) = [  3.27011E+16 0.02700  2.27467E-04 0.02698 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78396E+17 0.01271  1.24075E-03 0.01269 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001421 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06972E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001421 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494760 5.50002E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3575044 3.57833E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931617 9.32347E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001421 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.65775E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36673E+20 4.4E-06  2.36673E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35902E+19 9.8E-08  9.35902E+19 9.8E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43764E+20 0.00028  1.28145E+20 0.00018  1.56188E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37354E+20 0.00017  2.21736E+20 0.00011  1.56188E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61406E+20 0.00035  2.61406E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50642E+22 0.00025  3.32235E+22 0.00021  1.84070E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43727E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61727E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16764E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  7.85765E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.85765E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.90936E-01 0.04473 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.09838E-01 0.01076 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.69722E-04 0.02056 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.23477E+03 0.02878 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06770E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.08121E-01 0.02801 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.32798E-01 0.02801 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52882E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00069E+02 9.8E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04929E-01 0.00044  1.00200E-01 0.00044  3.47717E-04 0.00819 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05255E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05418E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05255E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98318E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.80990E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81142E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65473E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.65187E-02 0.00084 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10504E-01 0.00147 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10317E-01 0.00042 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93072E-03 0.00507  2.87074E-04 0.02005  9.16687E-04 0.01133  7.11531E-04 0.01197  1.57965E-03 0.00796  3.49537E-04 0.01838  8.62435E-05 0.03739 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.14133E-01 0.01520  1.24791E-02 7.6E-05  3.21963E-02 0.00017  1.06922E-01 0.00060  3.01445E-01 0.00031  1.25852E+00 0.00103  6.46333E+00 0.02496 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49436E-03 0.00725  2.54587E-04 0.03059  8.11696E-04 0.01638  6.33779E-04 0.01762  1.41015E-03 0.01232  3.04378E-04 0.02521  7.97727E-05 0.05347 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.18119E-01 0.02212  1.24790E-02 7.7E-05  3.22067E-02 0.00025  1.06874E-01 0.00079  3.01541E-01 0.00046  1.26029E+00 0.00125  6.94271E+00 0.02064 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.40390E-07 0.00132  8.39990E-07 0.00132  9.56267E-07 0.01580 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.60453E-07 0.00125  7.60091E-07 0.00126  8.65334E-07 0.01582 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46242E-03 0.00825  2.44272E-04 0.03765  8.04660E-04 0.01856  6.04000E-04 0.02089  1.41762E-03 0.01418  3.13344E-04 0.03041  7.85221E-05 0.06324 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.21787E-01 0.02700  1.24784E-02 0.00012  3.21922E-02 0.00029  1.06966E-01 0.00112  3.01323E-01 0.00051  1.25550E+00 0.00213  6.78182E+00 0.03065 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.39869E-07 0.01970  7.39476E-07 0.01970  8.12786E-07 0.05966 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.69750E-07 0.01970  6.69393E-07 0.01970  7.36095E-07 0.05978 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.39543E-03 0.04201  3.04794E-04 0.12571  6.91260E-04 0.08011  5.51880E-04 0.09613  1.45718E-03 0.06601  2.95243E-04 0.13150  9.50760E-05 0.21639 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.74194E-01 0.10662  1.24760E-02 0.00016  3.21562E-02 0.00104  1.06368E-01 0.00269  3.01533E-01 0.00192  1.25668E+00 0.00641  7.09559E+00 0.07275 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.40929E-03 0.04219  3.00085E-04 0.12570  6.93342E-04 0.07995  5.54325E-04 0.09699  1.47013E-03 0.06642  2.92829E-04 0.12905  9.85788E-05 0.21618 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.88424E-01 0.11145  1.24760E-02 0.00016  3.21542E-02 0.00105  1.06384E-01 0.00269  3.01538E-01 0.00189  1.25696E+00 0.00639  7.09749E+00 0.07277 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.60461E+03 0.03699 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.31250E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.52179E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.52401E-03 0.00536 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.23996E+03 0.00539 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.01686E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.83469E-05 0.00585  6.83658E-05 0.00586  1.07635E-06 0.44790 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.10377E-05 0.02067  6.09850E-05 0.02074  1.37684E-06 0.57649 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.66190E-04 0.01966  2.66739E-04 0.01970  1.27578E-04 0.44528 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62590E+01 0.01097 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46225E+01 0.00026  3.86297E+01 0.00038 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:26:46 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 05:14:04 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650054406187 ;
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

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02122E+00  9.89561E-01  9.99338E-01  9.99617E-01  1.00143E+00  9.92803E-01  9.97668E-01  9.96904E-01  1.00354E+00  9.93313E-01  9.93577E-01  9.95189E-01  1.00481E+00  9.98824E-01  1.00682E+00  1.00194E+00  9.97702E-01  1.00574E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.91256E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70874E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35235E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49615E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63145E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47675E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46734E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.31060E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.07533E+00 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334395 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33403E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33403E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.30428E+03 ;
RUNNING_TIME              (idx, 1)        =  7.07312E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.66917E-01  9.66917E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.04638E+00  1.46700E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.03715E+02  2.85917E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.57750E-01  1.80500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.01500E-02  1.33000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.24211E+01  3.97775E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.03374E+02  7.03374E+02 ];
CPU_USAGE                 (idx, 1)        = 8.91302 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98531E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.80943E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22048.28;
MEMSIZE                   (idx, 1)        = 21837.89;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 210.39;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.90633E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73704E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.58651E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90571E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32757E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96646E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57031E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65562E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61309E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47114E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16154E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78539E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43987E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03082E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33600E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30980E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51838E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04355E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13363E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21834E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01533E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.38494E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98670E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.85914E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16765E-01 -4.78609E+28  2.68657E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01316E+00 0.00078 ];
TH232_FISS                (idx, [1:   4]) = [  1.50318E+18 0.00434  1.60599E-02 0.00429 ];
U233_FISS                 (idx, [1:   4]) = [  7.92629E+19 0.00061  8.46862E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.07265E+18 0.00199  7.55652E-02 0.00190 ];
U238_FISS                 (idx, [1:   4]) = [  1.57818E+14 0.40485  1.68391E-06 0.40486 ];
PU239_FISS                (idx, [1:   4]) = [  6.58871E+17 0.00622  7.04022E-03 0.00624 ];
PU240_FISS                (idx, [1:   4]) = [  8.64193E+16 0.01747  9.23383E-04 0.01744 ];
PU241_FISS                (idx, [1:   4]) = [  1.53373E+17 0.01321  1.63894E-03 0.01325 ];
TH232_CAPT                (idx, [1:   4]) = [  9.19624E+19 0.00051  6.38783E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.14955E+19 0.00145  7.98502E-02 0.00141 ];
U235_CAPT                 (idx, [1:   4]) = [  2.47366E+18 0.00345  1.71823E-02 0.00343 ];
U238_CAPT                 (idx, [1:   4]) = [  2.83158E+15 0.09868  1.96850E-05 0.09868 ];
PU239_CAPT                (idx, [1:   4]) = [  3.02961E+17 0.00899  2.10433E-03 0.00896 ];
PU240_CAPT                (idx, [1:   4]) = [  2.25663E+17 0.01052  1.56752E-03 0.01053 ];
PU241_CAPT                (idx, [1:   4]) = [  3.11580E+16 0.02964  2.16402E-04 0.02960 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81837E+17 0.01131  1.26316E-03 0.01132 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002096 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08025E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002096 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5490570 5.49552E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3569722 3.57276E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 941804 9.42518E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002096 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.08616E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.7E-09  3.52483E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36687E+20 4.7E-06  2.36687E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35893E+19 9.9E-08  9.35893E+19 9.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43946E+20 0.00028  1.27914E+20 0.00019  1.60320E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37535E+20 0.00017  2.21503E+20 0.00011  1.60320E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61971E+20 0.00033  2.61971E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53970E+22 0.00024  3.35318E+22 0.00020  1.86510E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46923E+19 0.00150 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62227E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17152E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  7.69589E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.69589E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.29143E-01 0.04095 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.14386E-01 0.00952 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.30918E-04 0.01894 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.76971E+03 0.03399 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.05754E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.48020E-01 0.02430 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.68005E-01 0.02430 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52899E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 9.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03400E-01 0.00046  1.00049E-01 0.00046  3.48074E-04 0.00831 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.03582E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03512E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.03582E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97619E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81228E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81212E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65075E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.65073E-02 0.00086 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10895E-01 0.00140 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10815E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.95716E-03 0.00536  2.84804E-04 0.01942  9.18979E-04 0.01120  7.23695E-04 0.01200  1.57852E-03 0.00798  3.63073E-04 0.01828  8.80838E-05 0.03331 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.17244E-01 0.01390  1.24797E-02 8.1E-05  3.21968E-02 0.00016  1.06946E-01 0.00056  3.01275E-01 0.00031  1.25915E+00 0.00095  6.51281E+00 0.02372 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49446E-03 0.00834  2.49509E-04 0.02838  8.06228E-04 0.01665  6.34289E-04 0.01762  1.39984E-03 0.01190  3.26931E-04 0.02541  7.76625E-05 0.04684 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.18398E-01 0.02051  1.24801E-02 0.00010  3.22128E-02 0.00022  1.06917E-01 0.00081  3.01334E-01 0.00043  1.25963E+00 0.00119  6.83437E+00 0.02085 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.49332E-07 0.00151  8.48772E-07 0.00151  1.00898E-06 0.02246 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.67247E-07 0.00146  7.66741E-07 0.00147  9.11513E-07 0.02238 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46373E-03 0.00865  2.53926E-04 0.03192  8.15777E-04 0.01861  6.18220E-04 0.02119  1.39486E-03 0.01448  3.05602E-04 0.03179  7.53426E-05 0.06026 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04304E-01 0.02523  1.24792E-02 0.00011  3.21980E-02 0.00025  1.06860E-01 0.00109  3.01187E-01 0.00055  1.25740E+00 0.00178  6.61312E+00 0.03233 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.44910E-07 0.01984  7.44471E-07 0.01984  8.62315E-07 0.05586 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.73221E-07 0.01984  6.72824E-07 0.01983  7.79273E-07 0.05587 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.86174E-03 0.04433  1.91136E-04 0.13992  6.50727E-04 0.09005  4.81348E-04 0.09180  1.17509E-03 0.06231  3.10199E-04 0.13135  5.32384E-05 0.31125 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.09068E-01 0.11162  1.24833E-02 0.00056  3.21773E-02 0.00071  1.06709E-01 0.00332  3.01556E-01 0.00198  1.25865E+00 0.00560  7.11740E+00 0.09263 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87729E-03 0.04291  1.92347E-04 0.13934  6.53315E-04 0.08563  4.81458E-04 0.08827  1.18476E-03 0.06112  3.10404E-04 0.12789  5.50076E-05 0.30642 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.07441E-01 0.11057  1.24840E-02 0.00061  3.21774E-02 0.00072  1.06661E-01 0.00325  3.01512E-01 0.00196  1.25825E+00 0.00567  7.11980E+00 0.09254 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88287E+03 0.04097 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.41392E-07 0.00087 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.60061E-07 0.00071 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43331E-03 0.00598 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.08112E+03 0.00599 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.18523E-09 0.00106 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.85101E-05 0.00548  6.85160E-05 0.00548  1.03970E-06 0.46262 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.53754E-05 0.01745  6.54086E-05 0.01739  8.07230E-07 0.66644 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.25606E-04 0.01861  3.26390E-04 0.01867  1.26816E-04 0.44611 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63130E+01 0.01134 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46734E+01 0.00028  3.86547E+01 0.00038 ];

