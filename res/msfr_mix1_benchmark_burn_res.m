
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Jan 30 22:34:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02795E+00  9.94916E-01  9.80133E-01  9.95808E-01  1.00418E+00  9.92348E-01  1.00606E+00  9.98608E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.19592E-02 0.00097  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78041E-01 2.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.18189E-01 0.00047  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.32874E-01 0.00043  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67326E+00 0.00097  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.84105E+01 0.00067  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.84105E+01 0.00067  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.70230E+01 0.00080  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.26177E-02 0.00541  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000320 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00032E+04 0.00159 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00032E+04 0.00159 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.81050E+00 ;
RUNNING_TIME              (idx, 1)        =  1.89810E+00 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.57500E-02  2.57500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.01195E+00  1.01195E+00  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.89652E+00  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 4.64175 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.60472E+00 0.00172 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  5.51262E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  2.31036E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.81690E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.19014E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.31036E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.81690E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.22115E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.17886E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.22115E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.17886E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.33795E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.18915E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.48264E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.36343E+16 0.00098  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20806E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.08615E+00 0.00193 ];
TH232_FISS                (idx, [1:   4]) = [  1.64732E+18 0.01205  1.75391E-02 0.01189 ];
U233_FISS                 (idx, [1:   4]) = [  9.22646E+19 0.00134  9.82461E-01 0.00021 ];
TH232_CAPT                (idx, [1:   4]) = [  1.15183E+20 0.00148  8.07066E-01 0.00063 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37902E+19 0.00394  9.66342E-02 0.00394 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000320 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.23493E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000320 1.00123E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 603285 6.03855E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 397019 3.97364E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 16 1.60252E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000320 1.00123E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.22236E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36123E+20 8.9E-06  2.36123E+20 8.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37779E+19 3.8E-07  9.37779E+19 3.8E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42680E+20 0.00091  1.32151E+20 0.00072  1.05290E+19 0.00702 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36458E+20 0.00055  2.25929E+20 0.00042  1.05290E+19 0.00702 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36343E+20 0.00098  2.36343E+20 0.00098  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49624E+22 0.00078  3.37678E+22 0.00074  1.19457E+21 0.00670 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.78915E+15 0.26269 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.36462E+20 0.00055 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14545E+22 0.00096 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51078E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.09378E-01 0.22462 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21151E-01 0.02083 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.98438E-04 0.04180 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.03094E+03 0.00441 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 4.2E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  1.90816E-01 0.20753 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  1.90813E-01 0.20753 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51790E+00 8.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99669E+02 3.8E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.00048E+00 0.00117  9.97298E-01 0.00114  3.21076E-03 0.02719 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.99783E-01 0.00055 ];
COL_KEFF                  (idx, [1:   2]) = [  9.99167E-01 0.00098 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.99783E-01 0.00055 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99799E-01 0.00055 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.43338E+00 0.00053 ];
IMP_ALF                   (idx, [1:   2]) = [  7.43584E+00 0.00039 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  8.87479E-03 0.00402 ];
IMP_EALF                  (idx, [1:   2]) = [  8.84967E-03 0.00291 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.15102E-01 0.00426 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.14584E-01 0.00142 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.20867E-03 0.01813  2.40408E-04 0.06683  8.08201E-04 0.03405  5.92043E-04 0.04678  1.24219E-03 0.02677  2.76752E-04 0.06290  4.90732E-05 0.14045 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.32365E-01 0.04366  1.09778E-02 0.03711  3.23314E-02 0.00028  1.05201E-01 0.01029  2.97550E-01 0.00092  1.14933E+00 0.02758  2.12664E+00 0.15409 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.11388E-03 0.02674  2.20687E-04 0.09379  7.82416E-04 0.05015  5.45354E-04 0.06358  1.25912E-03 0.04637  2.57586E-04 0.08497  4.87138E-05 0.21461 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.33468E-01 0.06489  1.24751E-02 0.00014  3.23196E-02 0.00032  1.06274E-01 0.00263  2.97416E-01 0.00138  1.23589E+00 0.00082  5.40405E+00 0.09329 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.07750E-06 0.00322  1.07722E-06 0.00321  1.17845E-06 0.04818 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.07788E-06 0.00301  1.07760E-06 0.00300  1.17929E-06 0.04837 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20789E-03 0.02846  2.52038E-04 0.10002  7.97076E-04 0.05808  5.58982E-04 0.07715  1.24436E-03 0.04456  2.92584E-04 0.09607  6.28573E-05 0.19649 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.61666E-01 0.06939  1.24728E-02 0.00020  3.23337E-02 0.00053  1.06359E-01 0.00395  2.97472E-01 0.00147  1.23579E+00 0.00112  5.40011E+00 0.12595 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.57355E-07 0.03505  9.57933E-07 0.03506  9.01425E-07 0.16017 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.57428E-07 0.03501  9.58002E-07 0.03502  9.01905E-07 0.16044 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89253E-03 0.12769  1.64561E-04 0.43773  6.53063E-04 0.24312  5.85761E-04 0.26127  1.26344E-03 0.19879  2.18393E-04 0.30357  7.31191E-06 1.00000 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.13024E-01 0.13158  1.24695E-02 0.00080  3.23195E-02 0.00139  1.07916E-01 0.01391  2.95763E-01 0.00311  1.23317E+00 0.00330  3.29000E+00 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89968E-03 0.12613  1.65911E-04 0.41279  6.45878E-04 0.23459  6.12077E-04 0.26720  1.22992E-03 0.19299  2.35735E-04 0.31157  1.01605E-05 1.00000 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.12078E-01 0.13186  1.24695E-02 0.00080  3.23195E-02 0.00139  1.07916E-01 0.01391  2.95764E-01 0.00311  1.23317E+00 0.00330  3.29000E+00 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.99675E+03 0.12323 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.06935E-06 0.00176 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.06973E-06 0.00142 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32701E-03 0.02374 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.11288E+03 0.02380 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.97518E-09 0.00324 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.13817E-05 0.01584  7.13875E-05 0.01584  6.70672E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  8.35133E-05 0.05005  8.35485E-05 0.04999  2.32963E-07 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.13456E-04 0.04064  5.14114E-04 0.04058  2.57663E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.66366E+01 0.03554 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.84105E+01 0.00067  4.43204E+01 0.00092 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Jan 30 22:48:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01479E+00  9.99543E-01  9.86920E-01  9.98962E-01  1.00240E+00  1.00438E+00  9.96029E-01  9.96980E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.19249E-02 0.00086  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78075E-01 1.9E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.18148E-01 0.00047  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.32811E-01 0.00044  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67046E+00 0.00107  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.84827E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.84827E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.71947E+01 0.00074  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.26122E-02 0.00469  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000298 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00030E+04 0.00180 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00030E+04 0.00180 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.14846E+02 ;
RUNNING_TIME              (idx, 1)        =  1.58157E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.00667E-02  1.43167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.48083E+01  1.37964E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.97667E-02  9.97667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.79000E-02  1.79000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.58142E+01  3.04526E+01 ];
CPU_USAGE                 (idx, 1)        = 7.26152 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66709E+00 0.00027 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.44412E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 171 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.13609E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.63063E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.82815E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.84306E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.30309E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26093E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46826E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.93360E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.76965E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.87130E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  5.94725E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06229E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17492E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44428E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35395E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.16549E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54573E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.43796E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38056E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.61620E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53515E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.39038E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.03952E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.39067E+16 0.00092  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23373E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.09532E-04 -2.41853E+25  2.20830E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.10356E+00 0.00204 ];
TH232_FISS                (idx, [1:   4]) = [  1.63906E+18 0.01256  1.75038E-02 0.01230 ];
U233_FISS                 (idx, [1:   4]) = [  9.19527E+19 0.00159  9.82161E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  9.44812E+14 0.49251  9.97154E-06 0.49245 ];
TH232_CAPT                (idx, [1:   4]) = [  1.16558E+20 0.00138  7.99804E-01 0.00068 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37571E+19 0.00387  9.44015E-02 0.00375 ];
U235_CAPT                 (idx, [1:   4]) = [  4.69793E+14 0.70353  3.28090E-06 0.70359 ];
SM149_CAPT                (idx, [1:   4]) = [  2.05492E+16 0.09332  1.40963E-04 0.09316 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000298 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.22584E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000298 1.00123E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 609040 6.09595E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 391245 3.91618E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 13 1.30668E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000298 1.00123E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.36206E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36120E+20 8.5E-06  2.36120E+20 8.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37775E+19 3.6E-07  9.37775E+19 3.6E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45753E+20 0.00084  1.35076E+20 0.00063  1.06771E+19 0.00639 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39530E+20 0.00051  2.28853E+20 0.00037  1.06771E+19 0.00639 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.39067E+20 0.00092  2.39067E+20 0.00092  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54033E+22 0.00074  3.41951E+22 0.00069  1.20823E+21 0.00702 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.12596E+15 0.26001 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39533E+20 0.00051 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16038E+22 0.00094 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51171E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51171E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.97720E-01 0.16976 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06456E-01 0.02319 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.44178E-04 0.04928 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.88264E+03 0.01753 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99987E-01 3.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.95863E-01 0.15354 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.95859E-01 0.15354 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51787E+00 8.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99670E+02 3.6E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.85568E-01 0.00129  9.83095E-01 0.00129  3.03347E-03 0.02709 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.86975E-01 0.00051 ];
COL_KEFF                  (idx, [1:   2]) = [  9.87753E-01 0.00091 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.86975E-01 0.00051 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86988E-01 0.00051 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.43404E+00 0.00050 ];
IMP_ALF                   (idx, [1:   2]) = [  7.43661E+00 0.00039 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  8.86807E-03 0.00373 ];
IMP_EALF                  (idx, [1:   2]) = [  8.84293E-03 0.00293 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.17090E-01 0.00489 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.15904E-01 0.00134 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28240E-03 0.01987  2.47717E-04 0.05912  7.85966E-04 0.04289  6.23732E-04 0.04634  1.29774E-03 0.02773  2.78553E-04 0.05998  4.86975E-05 0.13425 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50234E-01 0.04488  1.21020E-02 0.01768  3.23587E-02 0.00038  1.05674E-01 0.00156  2.96950E-01 0.00080  1.18512E+00 0.02053  2.80488E+00 0.14335 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27564E-03 0.02441  2.42128E-04 0.08928  8.15105E-04 0.05215  6.67864E-04 0.06169  1.27126E-03 0.03895  2.46446E-04 0.09256  3.28362E-05 0.18071 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.07574E-01 0.04768  1.24764E-02 8.8E-05  3.23502E-02 0.00045  1.05631E-01 0.00194  2.97060E-01 0.00115  1.23492E+00 0.00087  6.86654E+00 0.07798 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.09728E-06 0.00355  1.09659E-06 0.00357  1.33465E-06 0.05175 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.08120E-06 0.00312  1.08053E-06 0.00315  1.31441E-06 0.05158 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.06241E-03 0.02776  1.78900E-04 0.11457  8.00171E-04 0.06324  5.81365E-04 0.06728  1.14914E-03 0.04479  2.93124E-04 0.09077  5.97081E-05 0.19293 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.86330E-01 0.07469  1.24780E-02 7.7E-05  3.23488E-02 0.00057  1.05519E-01 0.00244  2.96959E-01 0.00150  1.23507E+00 0.00114  6.60589E+00 0.11178 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.81632E-07 0.03539  9.80510E-07 0.03547  9.94593E-07 0.17291 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.67764E-07 0.03537  9.66656E-07 0.03544  9.79300E-07 0.17331 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.40931E-03 0.13546  1.39850E-04 0.60494  8.34993E-04 0.20948  4.35051E-04 0.23070  7.00204E-04 0.21660  2.85693E-04 0.42728  1.35227E-05 0.88094 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.59018E-01 0.17331  1.24680E-02 0.00091  3.23108E-02 0.00112  1.06875E-01 0.01146  2.94965E-01 0.00276  1.23154E+00 0.00345  1.02232E+01 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.45876E-03 0.12853  1.52360E-04 0.56531  8.17586E-04 0.20623  4.60019E-04 0.22612  7.21439E-04 0.20665  2.89452E-04 0.42407  1.79002E-05 0.77230 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.62175E-01 0.16133  1.24680E-02 0.00091  3.23108E-02 0.00112  1.06875E-01 0.01146  2.94965E-01 0.00276  1.23162E+00 0.00344  1.02232E+01 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.54867E+03 0.13481 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.09826E-06 0.00221 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.08220E-06 0.00167 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  2.99290E-03 0.02471 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.72928E+03 0.02502 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.92082E-09 0.00347 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.97736E-05 0.02170  6.97779E-05 0.02170  7.26600E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.73138E-05 0.05547  7.72190E-05 0.05541  2.03368E-06 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.63233E-04 0.04890  4.63762E-04 0.04864  2.29181E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60949E+01 0.03877 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.84827E+01 0.00069  4.43585E+01 0.00091 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Jan 30 23:04:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02443E+00  1.00409E+00  1.00392E+00  9.91544E-01  9.98507E-01  9.99444E-01  9.88879E-01  9.89191E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 2.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.18497E-02 0.00089  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78150E-01 2.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.18936E-01 0.00046  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.33533E-01 0.00043  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67528E+00 0.00095  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.83738E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.83738E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.66620E+01 0.00080  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.17332E-02 0.00591  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000358 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00036E+04 0.00177 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00036E+04 0.00177 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.32493E+02 ;
RUNNING_TIME              (idx, 1)        =  3.12564E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.60000E-02  1.59333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.01180E+01  1.53097E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.02467E-01  1.02700E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  3.64500E-02  1.85500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.12549E+01  1.98235E+02 ];
CPU_USAGE                 (idx, 1)        = 7.43826 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66646E+00 0.00028 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.70810E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.91985E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.71284E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.54369E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.32025E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.62977E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.56707E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51739E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.62904E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.62999E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.68844E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.93266E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.94038E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.63672E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.55163E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.41558E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.37511E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.60999E+18 ;
CS134_ACTIVITY            (idx, 1)        =  6.93710E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.21945E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.40809E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.62192E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41066E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.55490E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.41163E+16 0.00110  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45065E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.40134E-04 -1.19265E+26  2.20925E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.10572E+00 0.00199 ];
TH232_FISS                (idx, [1:   4]) = [  1.64501E+18 0.01309  1.75208E-02 0.01296 ];
U233_FISS                 (idx, [1:   4]) = [  9.20890E+19 0.00160  9.80912E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  2.09919E+16 0.10819  2.22799E-04 0.10795 ];
TH232_CAPT                (idx, [1:   4]) = [  1.16662E+20 0.00151  7.90520E-01 0.00068 ];
U233_CAPT                 (idx, [1:   4]) = [  1.38369E+19 0.00401  9.37692E-02 0.00401 ];
U235_CAPT                 (idx, [1:   4]) = [  6.75879E+15 0.19726  4.59926E-05 0.19748 ];
SM149_CAPT                (idx, [1:   4]) = [  8.42073E+16 0.05166  5.70606E-04 0.05182 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000358 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.23915E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000358 1.00124E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 611370 6.11937E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 388975 3.89289E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 13 1.30615E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000358 1.00124E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.72529E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36129E+20 8.9E-06  2.36129E+20 8.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 3.3E-07  9.37766E+19 3.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.47341E+20 0.00091  1.36622E+20 0.00073  1.07186E+19 0.00639 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.41118E+20 0.00055  2.30399E+20 0.00043  1.07186E+19 0.00639 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41163E+20 0.00110  2.41163E+20 0.00110  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.55737E+22 0.00082  3.43653E+22 0.00077  1.20839E+21 0.00603 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.13636E+15 0.30336 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41121E+20 0.00055 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16790E+22 0.00099 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51403E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51403E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  3.25591E-01 0.19271 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.71692E-01 0.03645 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.06764E-04 0.05934 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.52077E+03 0.04306 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99987E-01 4.0E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.53625E-01 0.16957 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.53620E-01 0.16957 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51799E+00 8.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 3.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.80384E-01 0.00128  9.77110E-01 0.00123  3.15290E-03 0.02434 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.80517E-01 0.00055 ];
COL_KEFF                  (idx, [1:   2]) = [  9.79241E-01 0.00109 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.80517E-01 0.00055 ];
ABS_KINF                  (idx, [1:   2]) = [  9.80530E-01 0.00055 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.41120E+00 0.00055 ];
IMP_ALF                   (idx, [1:   2]) = [  7.41372E+00 0.00043 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.07417E-03 0.00412 ];
IMP_EALF                  (idx, [1:   2]) = [  9.04847E-03 0.00321 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.18801E-01 0.00510 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.18735E-01 0.00133 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34301E-03 0.01530  2.72607E-04 0.05899  8.18001E-04 0.03805  6.27555E-04 0.03870  1.30386E-03 0.02939  2.75156E-04 0.06261  4.58323E-05 0.14564 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43701E-01 0.04837  1.16011E-02 0.02757  3.23328E-02 0.00026  1.06055E-01 0.00191  2.97698E-01 0.00092  1.16061E+00 0.02540  2.59239E+00 0.15213 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.34246E-03 0.02474  2.99820E-04 0.08762  8.47331E-04 0.05359  5.84009E-04 0.06256  1.28798E-03 0.04021  2.78880E-04 0.08542  4.44325E-05 0.23682 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.39784E-01 0.06470  1.24729E-02 0.00014  3.23356E-02 0.00038  1.05984E-01 0.00228  2.97590E-01 0.00137  1.23483E+00 0.00090  7.16727E+00 0.07733 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.08649E-06 0.00375  1.08582E-06 0.00381  1.26792E-06 0.04861 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.06506E-06 0.00366  1.06440E-06 0.00370  1.24431E-06 0.04888 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22877E-03 0.02471  2.94745E-04 0.08922  8.07861E-04 0.05820  5.43320E-04 0.06654  1.29888E-03 0.04800  2.48872E-04 0.12091  3.50905E-05 0.26778 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.39100E-01 0.08673  1.24754E-02 0.00014  3.23388E-02 0.00048  1.06188E-01 0.00360  2.97569E-01 0.00146  1.23379E+00 0.00135  8.88992E+00 0.08306 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.77102E-07 0.03540  9.75917E-07 0.03545  1.18819E-06 0.19485 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.58213E-07 0.03537  9.57043E-07 0.03542  1.16838E-06 0.19492 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.73217E-03 0.12029  1.72240E-04 0.40208  7.39985E-04 0.24712  3.08061E-04 0.34215  1.17442E-03 0.17938  3.05280E-04 0.42918  3.21866E-05 1.00000 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.79692E-01 0.38297  1.24794E-02 3.9E-09  3.23786E-02 0.00173  1.04645E-01 3.8E-09  2.96624E-01 0.00402  1.23028E+00 0.00483  1.02232E+01 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.66630E-03 0.11675  1.61764E-04 0.38204  6.85778E-04 0.24151  3.25927E-04 0.34983  1.17236E-03 0.16941  2.87691E-04 0.44792  3.27835E-05 1.00000 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.79598E-01 0.38294  1.24794E-02 5.6E-09  3.23841E-02 0.00173  1.04645E-01 3.8E-09  2.96551E-01 0.00391  1.23028E+00 0.00483  1.02232E+01 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.84866E+03 0.11723 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.08398E-06 0.00213 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.06254E-06 0.00169 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26215E-03 0.01473 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.01278E+03 0.01535 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.71714E-09 0.00292 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.66912E-05 0.03098  6.67294E-05 0.03097  4.16196E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.37025E-05 0.08032  6.36572E-05 0.08043  6.46405E-07 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.20699E-04 0.05845  3.20774E-04 0.05791  3.32061E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.78007E+01 0.03667 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.83738E+01 0.00069  4.41949E+01 0.00098 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Jan 30 23:19:54 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01699E+00  1.00754E+00  9.94723E-01  1.00089E+00  9.97823E-01  9.93366E-01  1.00011E+00  9.88560E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 2.6E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.17746E-02 0.00083  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78225E-01 1.8E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.19510E-01 0.00050  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.34046E-01 0.00047  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66482E+00 0.00099  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.82635E+01 0.00065  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.82635E+01 0.00065  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.62196E+01 0.00078  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.10928E-02 0.00532  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000599 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00060E+04 0.00200 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00060E+04 0.00200 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.52323E+02 ;
RUNNING_TIME              (idx, 1)        =  4.69810E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.23333E-02  1.63333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.57116E+01  1.55936E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.99483E-01  9.70167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  5.40833E-02  1.76333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.69794E+01  2.16633E+02 ];
CPU_USAGE                 (idx, 1)        = 7.49926 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66690E+00 0.00020 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.79759E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.04003E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73724E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.03659E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.32743E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.63301E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67983E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.54123E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.68008E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79991E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.80570E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.00039E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.87319E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.79950E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.50415E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.46407E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.41295E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.66070E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.26398E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.43841E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.48557E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.68570E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.68925E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.68435E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.40012E+16 0.00099  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28660E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.01339E-03 -2.23762E+26  2.21030E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.09353E+00 0.00215 ];
TH232_FISS                (idx, [1:   4]) = [  1.62977E+18 0.01161  1.73473E-02 0.01150 ];
U233_FISS                 (idx, [1:   4]) = [  9.20196E+19 0.00158  9.79431E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.57771E+16 0.04948  7.00568E-04 0.04953 ];
TH232_CAPT                (idx, [1:   4]) = [  1.14801E+20 0.00160  7.84442E-01 0.00074 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37220E+19 0.00413  9.37589E-02 0.00373 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49527E+16 0.10420  1.70602E-04 0.10414 ];
SM149_CAPT                (idx, [1:   4]) = [  1.25184E+17 0.04617  8.55622E-04 0.04618 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000599 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.22083E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000599 1.00122E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 609354 6.09744E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 391229 3.91461E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 16 1.59942E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000599 1.00122E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.56700E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36136E+20 9.4E-06  2.36136E+20 9.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 3.7E-07  9.37751E+19 3.7E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46368E+20 0.00086  1.35799E+20 0.00069  1.05687E+19 0.00699 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40143E+20 0.00052  2.29574E+20 0.00041  1.05687E+19 0.00699 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40012E+20 0.00099  2.40012E+20 0.00099  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.52728E+22 0.00081  3.40711E+22 0.00076  1.20174E+21 0.00751 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.84428E+15 0.29072 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.40146E+20 0.00052 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15969E+22 0.00100 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51407E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51407E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.04012E-01 0.15224 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31296E-01 0.03622 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.54892E-04 0.07192 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.55356E+03 0.04213 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 4.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.65817E-01 0.13117 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.65809E-01 0.13117 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51811E+00 9.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 3.7E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.85502E-01 0.00151  9.82661E-01 0.00150  3.12968E-03 0.02978 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.84517E-01 0.00052 ];
COL_KEFF                  (idx, [1:   2]) = [  9.83944E-01 0.00099 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.84517E-01 0.00052 ];
ABS_KINF                  (idx, [1:   2]) = [  9.84533E-01 0.00052 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.39130E+00 0.00054 ];
IMP_ALF                   (idx, [1:   2]) = [  7.39300E+00 0.00039 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.25630E-03 0.00404 ];
IMP_EALF                  (idx, [1:   2]) = [  9.23698E-03 0.00287 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.20837E-01 0.00483 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.20339E-01 0.00146 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34336E-03 0.01886  2.45570E-04 0.07016  8.17481E-04 0.03191  6.32179E-04 0.04003  1.33550E-03 0.02913  2.59953E-04 0.05837  5.26760E-05 0.13189 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.48585E-01 0.05019  1.14762E-02 0.02964  3.23586E-02 0.00038  1.05136E-01 0.01027  2.97630E-01 0.00087  1.14993E+00 0.02759  2.62978E+00 0.14184 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22565E-03 0.02684  2.56002E-04 0.09916  7.39968E-04 0.05035  6.13944E-04 0.06435  1.31490E-03 0.04338  2.43236E-04 0.08574  5.75965E-05 0.24194 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44420E-01 0.07366  1.24730E-02 0.00014  3.23366E-02 0.00034  1.06323E-01 0.00252  2.97752E-01 0.00112  1.23607E+00 0.00088  6.27995E+00 0.08077 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.05935E-06 0.00306  1.05911E-06 0.00305  1.13558E-06 0.05087 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.04383E-06 0.00289  1.04358E-06 0.00288  1.11890E-06 0.05093 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.17430E-03 0.03065  2.32171E-04 0.10714  7.07601E-04 0.04778  6.00434E-04 0.06788  1.32653E-03 0.04478  2.57840E-04 0.09739  4.97281E-05 0.23296 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.40789E-01 0.07481  1.24741E-02 0.00017  3.23589E-02 0.00069  1.06826E-01 0.00412  2.97332E-01 0.00139  1.23536E+00 0.00114  5.44700E+00 0.13840 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.33650E-07 0.03551  9.33836E-07 0.03552  8.84582E-07 0.15314 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.20830E-07 0.03557  9.21010E-07 0.03558  8.71454E-07 0.15279 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.81254E-03 0.12784  1.81417E-04 0.38888  7.40265E-04 0.20931  3.71202E-04 0.35709  1.25011E-03 0.19236  1.90414E-04 0.36179  7.91402E-05 0.60202 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.71112E-01 0.35125  1.24714E-02 0.00064  3.22745E-02 0.0E+00  1.05936E-01 0.00981  2.96451E-01 0.00385  1.23433E+00 0.00343  6.75662E+00 0.29622 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81542E-03 0.12497  1.75499E-04 0.36126  7.12496E-04 0.20948  3.90187E-04 0.33934  1.23312E-03 0.19305  2.05918E-04 0.35268  9.82080E-05 0.59615 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  5.74356E-01 0.34929  1.24714E-02 0.00064  3.22745E-02 0.0E+00  1.05880E-01 0.00973  2.96515E-01 0.00388  1.23433E+00 0.00343  6.75662E+00 0.29622 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.05432E+03 0.12418 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.04659E-06 0.00235 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.03116E-06 0.00165 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.05576E-03 0.01414 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.92237E+03 0.01463 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.60512E-09 0.00267 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.33584E-05 0.03893  6.33584E-05 0.03893  0.00000E+00 0.0E+00 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.30919E-05 0.07326  5.30919E-05 0.07326  0.00000E+00 0.0E+00 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.63875E-04 0.06970  2.64742E-04 0.06968  0.00000E+00 0.0E+00 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.57973E+01 0.03082 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.82635E+01 0.00065  4.40478E+01 0.00097 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Jan 30 23:36:23 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01954E+00  9.94102E-01  9.95990E-01  9.94567E-01  1.00009E+00  1.00455E+00  9.97006E-01  9.94160E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.17542E-02 0.00097  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78246E-01 2.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.20835E-01 0.00059  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.35330E-01 0.00055  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65363E+00 0.00106  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.81454E+01 0.00068  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.81454E+01 0.00068  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.54335E+01 0.00092  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.03189E-02 0.00587  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000224 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00022E+04 0.00186 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00022E+04 0.00186 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.77864E+02 ;
RUNNING_TIME              (idx, 1)        =  6.34663E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.02500E-02  1.79167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.20526E+01  1.63410E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.03067E-01  1.03583E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  7.17833E-02  1.77000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.34647E+01  2.20616E+02 ];
CPU_USAGE                 (idx, 1)        = 7.52942 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66538E+00 0.00031 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84324E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.08312E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75144E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.88592E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.29150E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.60474E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.75917E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55861E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.08816E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.94209E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.01433E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.85471E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.86069E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.95648E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.98558E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.50025E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.44168E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.69829E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.14282E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.85690E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.49768E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.73551E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.75045E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.68037E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.38328E+16 0.00120  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57321E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.90009E-03 -4.19552E+26  2.21226E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.07998E+00 0.00220 ];
TH232_FISS                (idx, [1:   4]) = [  1.63161E+18 0.01106  1.74219E-02 0.01090 ];
U233_FISS                 (idx, [1:   4]) = [  9.12847E+19 0.00150  9.74793E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  2.72441E+17 0.03270  2.90860E-03 0.03258 ];
TH232_CAPT                (idx, [1:   4]) = [  1.12149E+20 0.00180  7.73703E-01 0.00086 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37768E+19 0.00457  9.50492E-02 0.00442 ];
U235_CAPT                 (idx, [1:   4]) = [  9.44038E+16 0.04830  6.51594E-04 0.04839 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90119E+17 0.03371  1.31221E-03 0.03369 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000224 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15284E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000224 1.00115E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 607645 6.08189E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 392562 3.92947E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 17 1.69484E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000224 1.00115E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.49246E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36151E+20 1.1E-05  2.36151E+20 1.1E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37716E+19 4.1E-07  9.37716E+19 4.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44911E+20 0.00088  1.34346E+20 0.00062  1.05646E+19 0.00748 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38683E+20 0.00053  2.28118E+20 0.00036  1.05646E+19 0.00748 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38328E+20 0.00120  2.38328E+20 0.00120  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.48545E+22 0.00087  3.36433E+22 0.00075  1.21120E+21 0.00802 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.04779E+15 0.25151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38687E+20 0.00053 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14867E+22 0.00114 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51093E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51093E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.01976E-01 0.13977 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.18100E-01 0.03600 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.32845E-04 0.05964 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.62930E+03 0.03477 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99983E-01 4.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.04660E-01 0.12058 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.04654E-01 0.12058 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51836E+00 1.1E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99682E+02 4.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.89628E-01 0.00133  9.86402E-01 0.00136  3.15767E-03 0.02718 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.90584E-01 0.00053 ];
COL_KEFF                  (idx, [1:   2]) = [  9.91006E-01 0.00119 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.90584E-01 0.00053 ];
ABS_KINF                  (idx, [1:   2]) = [  9.90601E-01 0.00053 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.35509E+00 0.00053 ];
IMP_ALF                   (idx, [1:   2]) = [  7.35065E+00 0.00044 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.59709E-03 0.00387 ];
IMP_EALF                  (idx, [1:   2]) = [  9.63762E-03 0.00322 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.24821E-01 0.00500 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.24674E-01 0.00171 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.25560E-03 0.01896  2.97657E-04 0.06394  7.89043E-04 0.03130  5.89252E-04 0.04504  1.26767E-03 0.02824  2.66623E-04 0.06612  4.53547E-05 0.14989 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.37141E-01 0.05257  1.17264E-02 0.02539  3.23293E-02 0.00031  1.06219E-01 0.00240  2.97285E-01 0.00079  1.12532E+00 0.03162  2.35168E+00 0.15762 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24051E-03 0.02413  2.90157E-04 0.08181  7.98723E-04 0.04997  6.02302E-04 0.06622  1.23899E-03 0.03836  2.58791E-04 0.08459  5.15508E-05 0.21645 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45505E-01 0.07287  1.24763E-02 9.9E-05  3.23305E-02 0.00041  1.06226E-01 0.00322  2.97265E-01 0.00105  1.23685E+00 0.00113  6.26822E+00 0.08794 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.02423E-06 0.00360  1.02409E-06 0.00362  1.08416E-06 0.04610 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.01351E-06 0.00357  1.01337E-06 0.00360  1.07207E-06 0.04594 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18899E-03 0.02831  3.12818E-04 0.08913  7.16141E-04 0.04812  5.68727E-04 0.07665  1.30224E-03 0.04220  2.45193E-04 0.10979  4.38659E-05 0.22901 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.36963E-01 0.07928  1.24765E-02 0.00012  3.23151E-02 0.00045  1.06318E-01 0.00400  2.97131E-01 0.00124  1.23665E+00 0.00160  7.36837E+00 0.11577 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.05518E-07 0.03510  9.05157E-07 0.03511  8.15757E-07 0.13865 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.95932E-07 0.03512  8.95572E-07 0.03513  8.07535E-07 0.13913 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80832E-03 0.12160  5.08892E-04 0.35966  4.50202E-04 0.28589  4.49413E-04 0.27099  1.14395E-03 0.17794  2.05886E-04 0.40594  4.99769E-05 0.76692 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.17494E-01 0.35755  1.24794E-02 3.9E-09  3.23247E-02 0.00272  1.07612E-01 0.01243  2.96188E-01 0.00338  1.24763E+00 0.01130  1.02232E+01 9.1E-09 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.78109E-03 0.12199  5.04840E-04 0.34949  4.39578E-04 0.28622  4.47894E-04 0.27497  1.13347E-03 0.17615  1.97482E-04 0.38576  5.78279E-05 0.67457 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.32723E-01 0.34987  1.24794E-02 0.0E+00  3.23298E-02 0.00264  1.07703E-01 0.01253  2.96271E-01 0.00340  1.24763E+00 0.01130  1.02232E+01 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.03287E+03 0.11466 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.01569E-06 0.00182 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.00501E-06 0.00140 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.14608E-03 0.01634 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.09476E+03 0.01555 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.51567E-09 0.00234 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69767E-05 0.03235  6.56460E-05 0.03298  1.33063E-06 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.22235E-05 0.07363  6.20926E-05 0.07405  1.30935E-07 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.45950E-04 0.05689  2.45747E-04 0.05771  3.86615E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73320E+01 0.03640 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.81454E+01 0.00068  4.37435E+01 0.00094 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sun Jan 30 23:53:03 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01797E+00  1.00671E+00  1.00395E+00  9.95888E-01  9.94363E-01  9.94494E-01  9.94472E-01  9.92156E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 2.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.17418E-02 0.00088  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78258E-01 1.9E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.21279E-01 0.00050  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.35754E-01 0.00046  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63940E+00 0.00092  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.79502E+01 0.00062  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.79502E+01 0.00062  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.48652E+01 0.00076  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.07736E-02 0.00532  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000404 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00040E+04 0.00170 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00040E+04 0.00170 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.04811E+02 ;
RUNNING_TIME              (idx, 1)        =  8.01326E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.08267E-01  1.80167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.85754E+01  1.65228E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.00300E-01  9.72333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  8.99000E-02  1.81167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.01310E+01  2.28237E+02 ];
CPU_USAGE                 (idx, 1)        = 7.54762 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66583E+00 0.00022 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86978E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.03868E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74413E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.47609E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.24541E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.56948E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.76099E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55501E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.17442E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.00270E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.31953E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.67139E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.40766E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.03523E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.42714E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.47979E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.42675E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.67683E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.77197E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.23907E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.45094E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.72482E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.82812E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.57776E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.36323E+16 0.00093  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85981E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.81701E-03 -6.22013E+26  2.21428E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.06470E+00 0.00213 ];
TH232_FISS                (idx, [1:   4]) = [  1.58610E+18 0.01186  1.69382E-02 0.01187 ];
U233_FISS                 (idx, [1:   4]) = [  9.09083E+19 0.00154  9.70695E-01 0.00028 ];
U235_FISS                 (idx, [1:   4]) = [  5.14128E+17 0.02255  5.48917E-03 0.02247 ];
TH232_CAPT                (idx, [1:   4]) = [  1.09614E+20 0.00146  7.66833E-01 0.00080 ];
U233_CAPT                 (idx, [1:   4]) = [  1.35680E+19 0.00443  9.49188E-02 0.00428 ];
U235_CAPT                 (idx, [1:   4]) = [  2.09234E+17 0.03191  1.46422E-03 0.03198 ];
SM149_CAPT                (idx, [1:   4]) = [  1.89869E+17 0.03622  1.32728E-03 0.03589 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000404 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.18045E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000404 1.00118E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 604412 6.04868E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 395974 3.96294E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 18 1.79613E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000404 1.00118E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.91038E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36151E+20 1.1E-05  2.36151E+20 1.1E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37672E+19 3.8E-07  9.37672E+19 3.8E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43155E+20 0.00079  1.32737E+20 0.00056  1.04179E+19 0.00728 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36922E+20 0.00048  2.26504E+20 0.00033  1.04179E+19 0.00728 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36323E+20 0.00093  2.36323E+20 0.00093  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44050E+22 0.00075  3.32147E+22 0.00070  1.19035E+21 0.00739 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.25370E+15 0.26715 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.36926E+20 0.00048 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.13444E+22 0.00096 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50725E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50725E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.50343E-01 0.15989 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.97593E-01 0.04702 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94318E-04 0.06984 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.47251E+03 0.04562 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 4.8E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.08101E-01 0.14996 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.08095E-01 0.14996 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51848E+00 1.1E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99692E+02 3.8E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.98096E-01 0.00126  9.94914E-01 0.00125  3.23963E-03 0.02488 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.97939E-01 0.00048 ];
COL_KEFF                  (idx, [1:   2]) = [  9.99358E-01 0.00093 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.97939E-01 0.00048 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97957E-01 0.00048 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.31777E+00 0.00055 ];
IMP_ALF                   (idx, [1:   2]) = [  7.32274E+00 0.00043 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.96276E-03 0.00405 ];
IMP_EALF                  (idx, [1:   2]) = [  9.91011E-03 0.00316 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.29763E-01 0.00590 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.27603E-01 0.00160 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38539E-03 0.01761  2.59081E-04 0.06088  8.21907E-04 0.03393  6.15483E-04 0.04415  1.33776E-03 0.02582  2.99133E-04 0.05405  5.20260E-05 0.13243 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50388E-01 0.04183  1.16012E-02 0.02757  3.23111E-02 0.00026  1.06231E-01 0.00193  2.97678E-01 0.00088  1.19047E+00 0.02055  2.55428E+00 0.14613 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28829E-03 0.02415  2.56162E-04 0.09122  7.52452E-04 0.04871  5.82509E-04 0.05902  1.34528E-03 0.03666  3.02548E-04 0.09209  4.93394E-05 0.18788 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.52027E-01 0.04804  1.24751E-02 0.00011  3.23096E-02 0.00041  1.06338E-01 0.00281  2.97622E-01 0.00132  1.23973E+00 0.00146  6.30411E+00 0.08335 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.93083E-07 0.00309  9.92846E-07 0.00313  1.04147E-06 0.04977 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.91057E-07 0.00290  9.90811E-07 0.00290  1.04143E-06 0.05010 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26623E-03 0.02635  2.41035E-04 0.10508  7.46388E-04 0.05891  5.81833E-04 0.06758  1.33434E-03 0.04363  2.92422E-04 0.09568  7.02181E-05 0.18376 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.10352E-01 0.07703  1.24756E-02 0.00016  3.23220E-02 0.00070  1.06483E-01 0.00392  2.98065E-01 0.00171  1.24291E+00 0.00259  6.41581E+00 0.10591 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.82971E-07 0.03544  8.82587E-07 0.03545  7.73025E-07 0.11819 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.81211E-07 0.03541  8.80823E-07 0.03542  7.73059E-07 0.11831 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.21508E-03 0.11647  2.81559E-04 0.38595  8.92805E-04 0.24455  5.33139E-04 0.27920  1.28243E-03 0.17653  2.11706E-04 0.32588  1.34411E-05 1.00000 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.67833E-01 0.11692  1.24805E-02 9.0E-05  3.23234E-02 0.00151  1.05424E-01 0.00739  3.00085E-01 0.00528  1.23433E+00 0.00343  8.63638E+00 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.21479E-03 0.11474  2.92482E-04 0.37623  9.12489E-04 0.23991  5.30443E-04 0.28521  1.28011E-03 0.16566  1.89567E-04 0.31091  9.69838E-06 1.00000 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.54205E-01 0.11022  1.24805E-02 9.0E-05  3.23234E-02 0.00151  1.05424E-01 0.00739  2.99943E-01 0.00525  1.23433E+00 0.00343  8.63638E+00 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.71942E+03 0.11295 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.87364E-07 0.00208 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.85290E-07 0.00140 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20125E-03 0.01431 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.24224E+03 0.01433 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.42000E-09 0.00227 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.18928E-05 0.03734  6.18901E-05 0.03734  6.75534E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.40288E-05 0.08745  5.38024E-05 0.08762  1.10311E-06 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.04360E-04 0.06645  2.03997E-04 0.06697  2.96948E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69176E+01 0.03782 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.79502E+01 0.00062  4.33814E+01 0.00093 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 00:10:14 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01280E+00  9.91800E-01  1.00146E+00  9.89557E-01  1.00805E+00  9.98450E-01  9.99873E-01  9.98015E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 2.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.17934E-02 0.00093  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78207E-01 2.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.21779E-01 0.00052  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.36278E-01 0.00049  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63947E+00 0.00086  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.75484E+01 0.00072  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.75484E+01 0.00072  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.38501E+01 0.00093  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.00209E-02 0.00548  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000236 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00024E+04 0.00192 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00024E+04 0.00192 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.35633E+02 ;
RUNNING_TIME              (idx, 1)        =  9.73207E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.26983E-01  1.87167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  9.56136E+01  1.70382E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.98183E-01  9.78833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07650E-01  1.77500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  9.73191E+01  2.30434E+02 ];
CPU_USAGE                 (idx, 1)        = 7.55886 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66076E+00 0.00035 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88716E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 177 ;

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

TOT_ACTIVITY              (idx, 1)        =  7.04786E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75542E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.14855E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.19365E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.52881E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82225E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57070E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.34807E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.14287E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  6.25488E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.48749E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  7.17534E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.19313E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.21781E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.51002E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.44988E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.70751E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.57563E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.92011E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.44390E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.79490E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.81332E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.51553E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.34102E+16 0.00117  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43302E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -4.53489E-03 -1.00133E+27  2.21807E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.04276E+00 0.00192 ];
TH232_FISS                (idx, [1:   4]) = [  1.54023E+18 0.01115  1.64100E-02 0.01100 ];
U233_FISS                 (idx, [1:   4]) = [  9.02020E+19 0.00136  9.61103E-01 0.00028 ];
U235_FISS                 (idx, [1:   4]) = [  1.11696E+18 0.01320  1.19022E-02 0.01321 ];
TH232_CAPT                (idx, [1:   4]) = [  1.06139E+20 0.00178  7.55307E-01 0.00075 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34538E+19 0.00414  9.57386E-02 0.00379 ];
U235_CAPT                 (idx, [1:   4]) = [  4.16474E+17 0.02051  2.96267E-03 0.02031 ];
PU239_CAPT                (idx, [1:   4]) = [  2.38976E+14 1.00000  1.66694E-06 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85114E+17 0.03192  1.31694E-03 0.03173 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000236 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.19093E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000236 1.00119E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 599681 6.00252E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 400543 4.00927E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 12 1.19988E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000236 1.00119E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -9.77889E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.2E-09  3.52494E-02 4.2E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36143E+20 1.1E-05  2.36143E+20 1.1E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37577E+19 3.5E-07  9.37577E+19 3.5E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40423E+20 0.00084  1.30393E+20 0.00063  1.00300E+19 0.00671 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34181E+20 0.00051  2.24151E+20 0.00037  1.00300E+19 0.00671 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.34102E+20 0.00117  2.34102E+20 0.00117  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.38232E+22 0.00084  3.26725E+22 0.00079  1.15073E+21 0.00675 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.79878E+15 0.29559 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.34184E+20 0.00051 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11436E+22 0.00102 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49887E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49887E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.10961E-01 0.13711 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.76740E-01 0.02804 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.24151E-04 0.05982 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.45191E+03 0.04530 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99988E-01 3.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.04550E-01 0.12310 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.04543E-01 0.12310 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51865E+00 1.1E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99712E+02 3.5E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.01023E+00 0.00125  1.00658E+00 0.00124  3.20886E-03 0.02770 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.00959E+00 0.00051 ];
COL_KEFF                  (idx, [1:   2]) = [  1.00885E+00 0.00116 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.00959E+00 0.00051 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00960E+00 0.00051 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.26851E+00 0.00055 ];
IMP_ALF                   (idx, [1:   2]) = [  7.27223E+00 0.00044 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.04657E-02 0.00406 ];
IMP_EALF                  (idx, [1:   2]) = [  1.04238E-02 0.00324 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.33375E-01 0.00460 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.33936E-01 0.00158 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.25155E-03 0.01594  2.41612E-04 0.06667  7.77013E-04 0.03502  6.13445E-04 0.03540  1.27470E-03 0.02916  2.85587E-04 0.05901  5.91944E-05 0.13562 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.68294E-01 0.04708  1.14801E-02 0.02964  3.23037E-02 0.00037  1.06209E-01 0.00197  2.98093E-01 0.00099  1.19229E+00 0.02059  2.85471E+00 0.13288 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23099E-03 0.02415  2.54638E-04 0.09431  7.84316E-04 0.05122  5.80814E-04 0.05338  1.27046E-03 0.04253  2.72019E-04 0.08345  6.87450E-05 0.22624 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.75448E-01 0.08231  1.24781E-02 6.0E-05  3.22889E-02 0.00041  1.06131E-01 0.00225  2.97941E-01 0.00125  1.24108E+00 0.00173  6.70431E+00 0.06879 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.50988E-07 0.00350  9.50711E-07 0.00349  1.05175E-06 0.04323 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.60641E-07 0.00348  9.60363E-07 0.00348  1.06159E-06 0.04304 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.17792E-03 0.02881  2.28484E-04 0.11635  7.67694E-04 0.05840  5.77470E-04 0.05726  1.30442E-03 0.04964  2.45078E-04 0.10255  5.47746E-05 0.21598 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.64520E-01 0.07940  1.24781E-02 7.2E-05  3.22821E-02 0.00060  1.06351E-01 0.00340  2.98079E-01 0.00191  1.24172E+00 0.00270  7.02515E+00 0.10428 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.41385E-07 0.03521  8.40709E-07 0.03523  9.42053E-07 0.12296 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.50265E-07 0.03520  8.49585E-07 0.03521  9.52420E-07 0.12343 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.12012E-03 0.10371  3.31197E-04 0.42802  6.75031E-04 0.20730  4.39011E-04 0.26357  1.33226E-03 0.18286  2.91501E-04 0.34040  5.11135E-05 0.57027 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.78265E-01 0.31391  1.24794E-02 5.7E-09  3.22745E-02 0.0E+00  1.06904E-01 0.01203  2.97434E-01 0.00469  1.23433E+00 0.00343  6.75662E+00 0.29622 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09826E-03 0.10399  3.15408E-04 0.42657  6.80238E-04 0.20356  4.20623E-04 0.26867  1.33414E-03 0.17897  2.93466E-04 0.33740  5.43924E-05 0.60061 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.80214E-01 0.31288  1.24794E-02 8.0E-09  3.22745E-02 0.0E+00  1.06898E-01 0.01202  2.97508E-01 0.00468  1.23433E+00 0.00343  6.75662E+00 0.29622 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75292E+03 0.10008 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.46281E-07 0.00191 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.55797E-07 0.00130 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23578E-03 0.01832 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.41991E+03 0.01827 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.37569E-09 0.00250 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.31715E-05 0.04017  6.25162E-05 0.04182  6.55322E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.21782E-05 0.08298  5.20475E-05 0.08346  1.30624E-07 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.28186E-04 0.05912  2.27917E-04 0.05998  2.89057E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64562E+01 0.04263 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.75484E+01 0.00072  4.28665E+01 0.00100 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 00:27:48 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01902E+00  1.00626E+00  9.87991E-01  9.94511E-01  9.96478E-01  9.97879E-01  9.95287E-01  1.00257E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.9E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.18458E-02 0.00100  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78154E-01 2.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.22107E-01 0.00055  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.36638E-01 0.00051  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63705E+00 0.00097  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.70903E+01 0.00066  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.70903E+01 0.00066  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.27962E+01 0.00088  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.87925E-02 0.00638  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000191 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00019E+04 0.00139 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00019E+04 0.00139 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.69299E+02 ;
RUNNING_TIME              (idx, 1)        =  1.14881E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.47050E-01  2.00667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.13013E+02  1.73991E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.01050E-01  1.02867E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.25450E-01  1.78000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.14880E+02  2.35008E+02 ];
CPU_USAGE                 (idx, 1)        = 7.56693 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66469E+00 0.00022 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89920E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 179 ;

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

TOT_ACTIVITY              (idx, 1)        =  7.04686E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.76948E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.83937E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.11220E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.46413E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.90276E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.59133E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.84422E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.42587E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.70685E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.26385E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.57307E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.49629E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.36562E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.53333E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.46541E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.72932E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.36974E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.31345E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.42128E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.92660E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.87673E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.39791E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.30526E+16 0.00103  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28660E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -8.45593E-03 -1.86712E+27  2.22673E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02047E+00 0.00186 ];
TH232_FISS                (idx, [1:   4]) = [  1.52112E+18 0.01310  1.62548E-02 0.01303 ];
U233_FISS                 (idx, [1:   4]) = [  8.75880E+19 0.00154  9.36017E-01 0.00040 ];
U235_FISS                 (idx, [1:   4]) = [  2.76046E+18 0.00876  2.95011E-02 0.00867 ];
PU239_FISS                (idx, [1:   4]) = [  1.85192E+15 0.42167  1.98360E-05 0.42205 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00908E+20 0.00133  7.35490E-01 0.00082 ];
U233_CAPT                 (idx, [1:   4]) = [  1.30878E+19 0.00421  9.53918E-02 0.00403 ];
U235_CAPT                 (idx, [1:   4]) = [  9.95655E+17 0.01337  7.25755E-03 0.01339 ];
PU239_CAPT                (idx, [1:   4]) = [  9.29507E+14 0.49242  6.73622E-06 0.49244 ];
SM149_CAPT                (idx, [1:   4]) = [  2.07385E+17 0.03141  1.51174E-03 0.03137 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000191 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10485E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000191 1.00110E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 594616 5.95169E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 405561 4.05922E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 14 1.39899E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000191 1.00110E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.49246E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36069E+20 1.1E-05  2.36069E+20 1.1E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37333E+19 3.1E-07  9.37333E+19 3.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37162E+20 0.00078  1.27492E+20 0.00054  9.67025E+18 0.00757 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30895E+20 0.00046  2.21225E+20 0.00031  9.67025E+18 0.00757 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30526E+20 0.00103  2.30526E+20 0.00103  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.30461E+22 0.00067  3.19351E+22 0.00059  1.11096E+21 0.00733 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.22487E+15 0.24916 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30899E+20 0.00046 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08663E+22 0.00095 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47587E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47587E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.51243E-01 0.17745 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.87195E-01 0.04111 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.12186E-04 0.07779 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.40466E+03 0.05679 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99986E-01 3.5E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.85938E-01 0.16117 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.85937E-01 0.16117 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51852E+00 1.1E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99764E+02 3.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02219E+00 0.00107  1.01895E+00 0.00108  3.36549E-03 0.02685 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02361E+00 0.00047 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02415E+00 0.00103 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02361E+00 0.00047 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02363E+00 0.00047 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.19813E+00 0.00063 ];
IMP_ALF                   (idx, [1:   2]) = [  7.19902E+00 0.00035 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.12312E-02 0.00458 ];
IMP_EALF                  (idx, [1:   2]) = [  1.12132E-02 0.00250 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.46062E-01 0.00500 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.44584E-01 0.00142 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28124E-03 0.01711  2.81441E-04 0.06109  7.74903E-04 0.03445  6.13373E-04 0.04414  1.28106E-03 0.02252  2.62053E-04 0.06446  6.84155E-05 0.11566 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.90652E-01 0.05394  1.18534E-02 0.02306  3.22945E-02 0.00044  1.06673E-01 0.00216  2.99037E-01 0.00102  1.12151E+00 0.03355  3.44252E+00 0.11663 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.30089E-03 0.02614  2.69946E-04 0.08822  7.94543E-04 0.04820  6.27975E-04 0.05861  1.26989E-03 0.03575  2.51824E-04 0.09151  8.67141E-05 0.15888 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.12332E-01 0.06467  1.24768E-02 9.8E-05  3.22885E-02 0.00063  1.06440E-01 0.00250  2.98971E-01 0.00132  1.24691E+00 0.00205  6.71375E+00 0.06507 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.00783E-07 0.00397  9.00447E-07 0.00397  1.01169E-06 0.05495 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.20650E-07 0.00380  9.20308E-07 0.00380  1.03347E-06 0.05487 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27260E-03 0.02664  2.63898E-04 0.11041  8.06570E-04 0.05353  6.24886E-04 0.05218  1.24578E-03 0.04133  2.50330E-04 0.09330  8.11334E-05 0.17795 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.18712E-01 0.08651  1.24784E-02 8.5E-05  3.23256E-02 0.00092  1.06340E-01 0.00322  2.98557E-01 0.00152  1.24719E+00 0.00302  6.66579E+00 0.09109 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.88600E-07 0.03546  7.86890E-07 0.03551  9.32545E-07 0.13239 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.06358E-07 0.03554  8.04610E-07 0.03559  9.51608E-07 0.13210 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.26264E-03 0.12436  1.23276E-04 0.39846  9.52179E-04 0.22167  4.28787E-04 0.24522  1.34598E-03 0.21149  3.73379E-04 0.30086  3.90407E-05 0.80103 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.13005E-01 0.29273  1.24714E-02 0.00064  3.23166E-02 0.00161  1.06555E-01 0.00939  2.99028E-01 0.00496  1.24431E+00 0.00759  9.42980E+00 0.08414 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.28248E-03 0.12424  1.24198E-04 0.37483  9.88170E-04 0.22139  4.31598E-04 0.22968  1.36777E-03 0.21049  3.37422E-04 0.29688  3.33265E-05 0.86146 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.97644E-01 0.30017  1.24714E-02 0.00064  3.23171E-02 0.00160  1.06514E-01 0.00924  2.99126E-01 0.00503  1.24431E+00 0.00759  9.42980E+00 0.08414 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.22744E+03 0.12315 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.93748E-07 0.00216 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.13470E-07 0.00185 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42579E-03 0.01865 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.83282E+03 0.01859 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.25289E-09 0.00263 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  5.60432E-05 0.05206  5.60644E-05 0.05206  5.47151E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  4.82666E-05 0.09127  4.83008E-05 0.09122  3.66561E-07 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.19210E-04 0.07680  2.18900E-04 0.07618  3.39978E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.82760E+01 0.03961 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.70903E+01 0.00066  4.21573E+01 0.00113 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 00:45:45 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02215E+00  1.00670E+00  9.90566E-01  9.96947E-01  9.94029E-01  9.94479E-01  9.94290E-01  1.00084E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.19461E-02 0.00080  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78054E-01 1.8E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.22845E-01 0.00048  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37424E-01 0.00045  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63587E+00 0.00085  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.67404E+01 0.00064  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.67404E+01 0.00064  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.17828E+01 0.00084  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.89758E-02 0.00487  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000262 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00026E+04 0.00175 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00026E+04 0.00175 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.00592E+03 ;
RUNNING_TIME              (idx, 1)        =  1.32829E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.67550E-01  2.05000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.30792E+02  1.77796E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  8.04533E-01  1.03483E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.43067E-01  1.76167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.32827E+02  2.37966E+02 ];
CPU_USAGE                 (idx, 1)        = 7.57302 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66709E+00 0.00024 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90796E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 181 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.96423E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75913E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.31742E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01332E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38832E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.91937E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.58898E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.34868E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.85148E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.24939E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.18467E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.06742E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.92674E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.43172E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.47345E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.41466E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.66296E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.63564E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.35314E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.33075E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.00634E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.16289E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.18005E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.29191E+16 0.00105  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57638E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.69752E-02 -3.74823E+27  2.24554E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.99739E-01 0.00215 ];
TH232_FISS                (idx, [1:   4]) = [  1.43754E+18 0.01222  1.53262E-02 0.01207 ];
U233_FISS                 (idx, [1:   4]) = [  8.46340E+19 0.00147  9.02416E-01 0.00047 ];
U235_FISS                 (idx, [1:   4]) = [  5.06574E+18 0.00639  5.40098E-02 0.00612 ];
PU239_FISS                (idx, [1:   4]) = [  2.83496E+16 0.08933  3.02913E-04 0.08947 ];
PU240_FISS                (idx, [1:   4]) = [  1.82806E+15 0.34086  1.93981E-05 0.34090 ];
PU241_FISS                (idx, [1:   4]) = [  1.59999E+15 0.36648  1.70197E-05 0.36644 ];
TH232_CAPT                (idx, [1:   4]) = [  9.60137E+19 0.00176  7.07789E-01 0.00080 ];
U233_CAPT                 (idx, [1:   4]) = [  1.25637E+19 0.00396  9.26273E-02 0.00395 ];
U235_CAPT                 (idx, [1:   4]) = [  1.84562E+18 0.01058  1.36036E-02 0.01035 ];
PU239_CAPT                (idx, [1:   4]) = [  1.25796E+16 0.11334  9.28846E-05 0.11353 ];
PU240_CAPT                (idx, [1:   4]) = [  2.29763E+15 0.30157  1.70810E-05 0.30159 ];
PU241_CAPT                (idx, [1:   4]) = [  2.24531E+14 1.00000  1.70097E-06 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90441E+17 0.03321  1.40382E-03 0.03321 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000262 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09427E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000262 1.00109E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 591336 5.91858E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 408906 4.09216E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 20 2.00089E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000262 1.00109E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.70435E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.2E-09  3.52494E-02 4.2E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35956E+20 1.4E-05  2.35956E+20 1.4E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36943E+19 3.3E-07  9.36943E+19 3.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35698E+20 0.00066  1.26042E+20 0.00047  9.65651E+18 0.00619 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29393E+20 0.00039  2.19736E+20 0.00027  9.65651E+18 0.00619 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.29191E+20 0.00105  2.29191E+20 0.00105  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.26455E+22 0.00072  3.15227E+22 0.00066  1.12282E+21 0.00634 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.60521E+15 0.26506 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.29397E+20 0.00039 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07233E+22 0.00089 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.43964E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.43964E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.90865E-01 0.16677 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.16193E-01 0.03543 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.25194E-04 0.07052 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.89760E+03 0.03184 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99980E-01 5.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.30908E-01 0.14652 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.30905E-01 0.14652 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51836E+00 1.4E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99847E+02 3.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.03119E+00 0.00125  1.02701E+00 0.00120  3.56624E-03 0.02429 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02976E+00 0.00039 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02963E+00 0.00105 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02976E+00 0.00039 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02978E+00 0.00039 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.12366E+00 0.00066 ];
IMP_ALF                   (idx, [1:   2]) = [  7.12577E+00 0.00044 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.21001E-02 0.00462 ];
IMP_EALF                  (idx, [1:   2]) = [  1.20675E-02 0.00309 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.57677E-01 0.00469 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.57355E-01 0.00171 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41263E-03 0.01670  2.73190E-04 0.06092  8.30449E-04 0.03183  6.20650E-04 0.03568  1.32444E-03 0.02838  2.96073E-04 0.05390  6.78329E-05 0.11855 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05549E-01 0.04786  1.17258E-02 0.02539  3.22642E-02 0.00041  1.06556E-01 0.00162  2.99757E-01 0.00102  1.19253E+00 0.02314  4.09492E+00 0.10625 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.40281E-03 0.02140  2.74990E-04 0.08717  7.97359E-04 0.04261  6.29029E-04 0.05914  1.32082E-03 0.03978  3.04662E-04 0.07288  7.59556E-05 0.18108 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.19943E-01 0.07182  1.24737E-02 0.00013  3.22709E-02 0.00049  1.06655E-01 0.00231  2.99787E-01 0.00136  1.25636E+00 0.00247  7.64337E+00 0.05270 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.62487E-07 0.00369  8.62139E-07 0.00368  9.52470E-07 0.05327 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.89252E-07 0.00350  8.88889E-07 0.00348  9.82693E-07 0.05354 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44200E-03 0.02492  2.93289E-04 0.09034  8.47049E-04 0.05517  6.50502E-04 0.05635  1.33178E-03 0.03853  2.61742E-04 0.08429  5.76367E-05 0.19288 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.67831E-01 0.07618  1.24762E-02 0.00013  3.22832E-02 0.00072  1.06336E-01 0.00240  3.00022E-01 0.00150  1.25469E+00 0.00351  7.11367E+00 0.09436 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.66298E-07 0.03486  7.66300E-07 0.03486  6.92935E-07 0.12401 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.90327E-07 0.03488  7.90337E-07 0.03488  7.12099E-07 0.12357 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14647E-03 0.09989  3.08559E-04 0.32147  8.02075E-04 0.24849  4.82963E-04 0.30920  1.42046E-03 0.15224  1.32407E-04 0.44187  0.00000E+00 0.0E+00 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.48444E-01 0.12091  1.24741E-02 0.00046  3.23034E-02 0.00232  1.07469E-01 0.01057  3.01247E-01 0.00500  1.26203E+00 0.01358  0.00000E+00 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.14341E-03 0.09650  2.88584E-04 0.30981  8.68604E-04 0.22294  4.50629E-04 0.29349  1.40755E-03 0.15303  1.28050E-04 0.44636  0.00000E+00 0.0E+00 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.46906E-01 0.12206  1.24742E-02 0.00046  3.23013E-02 0.00231  1.07697E-01 0.01094  3.01047E-01 0.00489  1.26231E+00 0.01349  0.00000E+00 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.16144E+03 0.09458 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.55021E-07 0.00162 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.81568E-07 0.00123 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37218E-03 0.01263 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.94583E+03 0.01283 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17597E-09 0.00259 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.29344E-05 0.04050  6.29344E-05 0.04050  0.00000E+00 0.0E+00 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.12543E-05 0.07926  5.12543E-05 0.07926  0.00000E+00 0.0E+00 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.37201E-04 0.06801  2.38011E-04 0.06799  0.00000E+00 0.0E+00 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75612E+01 0.03420 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.67404E+01 0.00064  4.14106E+01 0.00112 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 01:04:03 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01858E+00  9.96953E-01  9.95741E-01  9.95973E-01  9.93048E-01  1.00559E+00  9.98869E-01  9.95247E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 2.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.19557E-02 0.00082  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78044E-01 1.8E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.22698E-01 0.00045  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37288E-01 0.00042  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64877E+00 0.00093  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.66102E+01 0.00075  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.66102E+01 0.00075  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.15824E+01 0.00086  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.86631E-02 0.00511  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000349 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00035E+04 0.00198 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00035E+04 0.00198 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.14526E+03 ;
RUNNING_TIME              (idx, 1)        =  1.51137E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.88750E-01  2.12000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.48928E+02  1.81353E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.07317E-01  1.02783E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.60900E-01  1.78333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.51136E+02  2.40654E+02 ];
CPU_USAGE                 (idx, 1)        = 7.57757 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66599E+00 0.00026 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91447E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.86332E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73475E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.31797E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96572E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35466E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86628E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56818E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.38484E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99524E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.20427E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20909E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.14514E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.06735E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.86344E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.39773E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35580E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.58312E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.63555E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.77910E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24443E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.95366E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.52151E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.02258E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.28932E+16 0.00106  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21968E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.20010E-02 -4.85795E+27  2.25664E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00334E+00 0.00210 ];
TH232_FISS                (idx, [1:   4]) = [  1.42897E+18 0.01341  1.52724E-02 0.01321 ];
U233_FISS                 (idx, [1:   4]) = [  8.33616E+19 0.00165  8.91061E-01 0.00055 ];
U235_FISS                 (idx, [1:   4]) = [  5.74706E+18 0.00637  6.14260E-02 0.00605 ];
PU239_FISS                (idx, [1:   4]) = [  6.70291E+16 0.05535  7.16400E-04 0.05526 ];
PU240_FISS                (idx, [1:   4]) = [  1.83487E+15 0.34089  1.95886E-05 0.34087 ];
PU241_FISS                (idx, [1:   4]) = [  4.82143E+15 0.20626  5.17001E-05 0.20612 ];
TH232_CAPT                (idx, [1:   4]) = [  9.50077E+19 0.00155  7.00492E-01 0.00088 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23482E+19 0.00480  9.10388E-02 0.00450 ];
U235_CAPT                 (idx, [1:   4]) = [  2.08452E+18 0.00981  1.53685E-02 0.00968 ];
U238_CAPT                 (idx, [1:   4]) = [  4.66853E+14 0.70370  3.35122E-06 0.70362 ];
PU239_CAPT                (idx, [1:   4]) = [  3.36027E+16 0.08738  2.47466E-04 0.08761 ];
PU240_CAPT                (idx, [1:   4]) = [  7.12028E+15 0.18713  5.25695E-05 0.18728 ];
PU241_CAPT                (idx, [1:   4]) = [  6.80254E+14 0.57155  5.08342E-06 0.57155 ];
SM149_CAPT                (idx, [1:   4]) = [  2.02035E+17 0.03659  1.48992E-03 0.03664 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000349 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12932E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000349 1.00113E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 591983 5.92456E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 408353 4.08660E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 13 1.30026E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000349 1.00113E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -9.89530E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35954E+20 1.3E-05  2.35954E+20 1.3E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36799E+19 3.1E-07  9.36799E+19 3.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35631E+20 0.00074  1.26085E+20 0.00052  9.54580E+18 0.00632 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29311E+20 0.00044  2.19765E+20 0.00030  9.54580E+18 0.00632 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.28932E+20 0.00106  2.28932E+20 0.00106  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.25603E+22 0.00070  3.14502E+22 0.00065  1.11009E+21 0.00608 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.99912E+15 0.28085 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.29314E+20 0.00044 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06812E+22 0.00088 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42918E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42918E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.40328E-01 0.16988 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30287E-01 0.03646 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.03759E-04 0.05468 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.77686E+03 0.03924 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99987E-01 3.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.98823E-01 0.15727 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.98823E-01 0.15727 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51873E+00 1.2E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99878E+02 3.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02865E+00 0.00145  1.02604E+00 0.00134  3.39404E-03 0.02434 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.03014E+00 0.00043 ];
COL_KEFF                  (idx, [1:   2]) = [  1.03079E+00 0.00105 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.03014E+00 0.00043 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03016E+00 0.00043 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.09129E+00 0.00065 ];
IMP_ALF                   (idx, [1:   2]) = [  7.10456E+00 0.00039 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.24979E-02 0.00461 ];
IMP_EALF                  (idx, [1:   2]) = [  1.23251E-02 0.00281 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.66273E-01 0.00489 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.62101E-01 0.00149 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.35985E-03 0.01686  2.61641E-04 0.05940  7.95280E-04 0.03456  6.00139E-04 0.04156  1.34193E-03 0.03041  2.92909E-04 0.06416  6.79542E-05 0.12412 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.79824E-01 0.04299  1.19788E-02 0.02052  3.22539E-02 0.00044  1.06211E-01 0.00166  3.00552E-01 0.00102  1.16695E+00 0.02769  3.18397E+00 0.11859 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35704E-03 0.02482  2.48503E-04 0.09942  7.78560E-04 0.04953  6.13193E-04 0.06093  1.34551E-03 0.03969  3.15174E-04 0.08371  5.60945E-05 0.17064 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.77172E-01 0.05753  1.24782E-02 5.6E-05  3.22687E-02 0.00055  1.06383E-01 0.00245  3.00331E-01 0.00130  1.25413E+00 0.00280  6.49082E+00 0.06557 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.44327E-07 0.00345  8.43947E-07 0.00343  9.74106E-07 0.05971 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.68325E-07 0.00310  8.67936E-07 0.00308  1.00139E-06 0.05958 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27920E-03 0.02393  2.74071E-04 0.10033  7.95440E-04 0.06076  5.84544E-04 0.07013  1.25638E-03 0.04167  3.13513E-04 0.09810  5.52525E-05 0.20495 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51707E-01 0.06457  1.24755E-02 0.00015  3.22514E-02 0.00073  1.06782E-01 0.00380  3.00277E-01 0.00169  1.25964E+00 0.00437  5.76567E+00 0.11422 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.43550E-07 0.03469  7.43082E-07 0.03467  6.82600E-07 0.14507 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.65436E-07 0.03465  7.64959E-07 0.03464  7.02554E-07 0.14557 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01167E-03 0.10683  2.87872E-04 0.36041  1.00089E-03 0.21036  5.72488E-04 0.25601  8.05873E-04 0.18797  2.99702E-04 0.31898  4.48517E-05 0.63130 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.29947E-01 0.18160  1.24811E-02 8.1E-05  3.22414E-02 0.00183  1.06482E-01 0.00774  3.01339E-01 0.00550  1.26180E+00 0.01046  5.07213E+00 0.35136 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99127E-03 0.10834  2.82141E-04 0.38103  1.03743E-03 0.20403  5.89295E-04 0.25362  7.55831E-04 0.18797  2.75683E-04 0.32059  5.08863E-05 0.61918 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.34511E-01 0.18194  1.24811E-02 8.1E-05  3.22356E-02 0.00179  1.06462E-01 0.00774  3.01369E-01 0.00551  1.26175E+00 0.01047  5.07213E+00 0.35136 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.07243E+03 0.10266 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.40617E-07 0.00209 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.64505E-07 0.00138 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34416E-03 0.01455 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.98005E+03 0.01471 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16375E-09 0.00211 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.55655E-05 0.03691  6.55655E-05 0.03691  0.00000E+00 0.0E+00 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.31460E-05 0.07703  6.31460E-05 0.07703  0.00000E+00 0.0E+00 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.10680E-04 0.05360  2.11389E-04 0.05360  0.00000E+00 0.0E+00 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74229E+01 0.03946 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.66102E+01 0.00075  4.11377E+01 0.00118 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 01:22:43 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.01882E+00  9.91369E-01  1.00039E+00  1.00012E+00  9.89947E-01  1.00535E+00  9.93242E-01  1.00077E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.19978E-02 0.00094  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78002E-01 2.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.22772E-01 0.00048  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37390E-01 0.00044  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64625E+00 0.00092  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.64533E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.64533E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.12325E+01 0.00080  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.81405E-02 0.00649  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000362 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00036E+04 0.00192 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00036E+04 0.00192 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.28729E+03 ;
RUNNING_TIME              (idx, 1)        =  1.69795E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.09883E-01  2.11333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.67413E+02  1.84850E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.00442E+00  9.71000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.78800E-01  1.79000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.69794E+02  2.42808E+02 ];
CPU_USAGE                 (idx, 1)        = 7.58145 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66701E+00 0.00023 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91937E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 185 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.85863E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73324E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.09785E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95314E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34589E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87428E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56769E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.57259E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17288E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.31483E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.40700E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.22027E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.22479E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.24339E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38350E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34380E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56740E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.64786E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.15694E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23343E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96932E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.83645E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00034E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.29234E+16 0.00095  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86299E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.71413E-02 -5.99297E+27  2.26799E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.96134E-01 0.00236 ];
TH232_FISS                (idx, [1:   4]) = [  1.41121E+18 0.01233  1.50689E-02 0.01235 ];
U233_FISS                 (idx, [1:   4]) = [  8.26552E+19 0.00185  8.82474E-01 0.00051 ];
U235_FISS                 (idx, [1:   4]) = [  6.32403E+18 0.00556  6.75210E-02 0.00534 ];
PU239_FISS                (idx, [1:   4]) = [  1.12225E+17 0.05001  1.19708E-03 0.04999 ];
PU240_FISS                (idx, [1:   4]) = [  6.38067E+15 0.16844  6.83038E-05 0.16842 ];
PU241_FISS                (idx, [1:   4]) = [  5.00621E+15 0.22927  5.38568E-05 0.22921 ];
TH232_CAPT                (idx, [1:   4]) = [  9.40168E+19 0.00154  6.92197E-01 0.00098 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23828E+19 0.00424  9.11681E-02 0.00406 ];
U235_CAPT                 (idx, [1:   4]) = [  2.25505E+18 0.01033  1.66044E-02 0.01035 ];
PU239_CAPT                (idx, [1:   4]) = [  5.40478E+16 0.06366  3.97825E-04 0.06345 ];
PU240_CAPT                (idx, [1:   4]) = [  1.88334E+16 0.11202  1.38662E-04 0.11174 ];
PU241_CAPT                (idx, [1:   4]) = [  1.59471E+15 0.36634  1.17943E-05 0.36639 ];
SM149_CAPT                (idx, [1:   4]) = [  1.89186E+17 0.03217  1.39252E-03 0.03207 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000362 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11792E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000362 1.00112E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 592068 5.92522E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 408280 4.08582E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 14 1.38780E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000362 1.00112E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.02445E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35970E+20 1.3E-05  2.35970E+20 1.3E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36667E+19 3.0E-07  9.36667E+19 3.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35430E+20 0.00074  1.25944E+20 0.00055  9.48578E+18 0.00682 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29097E+20 0.00044  2.19611E+20 0.00031  9.48578E+18 0.00682 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.29234E+20 0.00095  2.29234E+20 0.00095  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.25182E+22 0.00070  3.14188E+22 0.00063  1.09939E+21 0.00736 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.19553E+15 0.33615 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.29100E+20 0.00044 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06600E+22 0.00094 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.41850E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.41850E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.82099E-01 0.16412 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.80337E-01 0.02985 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94868E-04 0.06587 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.76136E+03 0.03868 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99986E-01 4.7E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.29218E-01 0.14653 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.29215E-01 0.14653 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51925E+00 1.2E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99906E+02 3.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02908E+00 0.00139  1.02588E+00 0.00134  3.40003E-03 0.02798 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.03116E+00 0.00044 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02948E+00 0.00094 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.03116E+00 0.00044 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03118E+00 0.00044 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.08059E+00 0.00063 ];
IMP_ALF                   (idx, [1:   2]) = [  7.07871E+00 0.00037 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.26315E-02 0.00444 ];
IMP_EALF                  (idx, [1:   2]) = [  1.26472E-02 0.00266 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.66108E-01 0.00428 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.65762E-01 0.00148 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41207E-03 0.01529  2.69013E-04 0.05682  7.79910E-04 0.03525  6.18218E-04 0.03740  1.38160E-03 0.02454  2.97411E-04 0.06597  6.59198E-05 0.12365 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08214E-01 0.04900  1.19773E-02 0.02052  3.22473E-02 0.00044  1.06265E-01 0.00170  3.00706E-01 0.00097  1.15012E+00 0.03175  3.83303E+00 0.11386 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.36541E-03 0.02163  2.31414E-04 0.08740  7.83747E-04 0.04773  6.30758E-04 0.05377  1.35321E-03 0.03604  2.91399E-04 0.09052  7.48776E-05 0.17037 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.38959E-01 0.07087  1.24766E-02 8.7E-05  3.22432E-02 0.00055  1.06043E-01 0.00184  3.01113E-01 0.00134  1.26418E+00 0.00330  7.96201E+00 0.04957 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.38191E-07 0.00359  8.37834E-07 0.00361  9.74971E-07 0.05167 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.62365E-07 0.00318  8.61995E-07 0.00319  1.00366E-06 0.05172 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28900E-03 0.02856  2.60159E-04 0.10286  7.53958E-04 0.06022  5.48795E-04 0.06156  1.34233E-03 0.04296  3.09990E-04 0.08607  7.37614E-05 0.19605 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.40029E-01 0.08918  1.24776E-02 9.0E-05  3.22786E-02 0.00065  1.06196E-01 0.00249  3.01404E-01 0.00176  1.26220E+00 0.00589  7.72331E+00 0.08056 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.47164E-07 0.03448  7.46570E-07 0.03448  7.32931E-07 0.15061 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.68754E-07 0.03439  7.68142E-07 0.03439  7.53174E-07 0.15013 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.67078E-03 0.10889  2.74963E-04 0.34877  5.42551E-04 0.23020  2.93359E-04 0.30176  1.29962E-03 0.16120  2.11153E-04 0.39154  4.91325E-05 0.54572 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.33196E-01 0.21643  1.24794E-02 3.5E-05  3.22085E-02 0.00142  1.05796E-01 0.00525  3.01678E-01 0.00521  1.26508E+00 0.01248  8.54467E+00 0.20508 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.74832E-03 0.11063  3.01255E-04 0.35577  5.30531E-04 0.23182  2.95170E-04 0.30817  1.35663E-03 0.16422  2.19778E-04 0.38419  4.49499E-05 0.52401 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.28999E-01 0.20536  1.24793E-02 4.7E-05  3.22099E-02 0.00145  1.05765E-01 0.00506  3.01554E-01 0.00520  1.26508E+00 0.01248  8.54467E+00 0.20508 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.59537E+03 0.10337 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.33791E-07 0.00192 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.57859E-07 0.00122 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30502E-03 0.01994 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.96436E+03 0.01987 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.09153E-09 0.00229 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.27821E-05 0.03909  6.27821E-05 0.03909  0.00000E+00 0.0E+00 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.69772E-05 0.07976  5.69772E-05 0.07976  0.00000E+00 0.0E+00 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.03771E-04 0.06503  2.04472E-04 0.06503  0.00000E+00 0.0E+00 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67252E+01 0.03641 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.64533E+01 0.00069  4.09806E+01 0.00111 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 01:41:54 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02135E+00  9.98884E-01  1.00154E+00  9.98100E-01  9.89039E-01  1.00223E+00  9.91435E-01  9.97425E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.20076E-02 0.00094  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77992E-01 2.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.22631E-01 0.00048  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37253E-01 0.00044  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64675E+00 0.00095  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.63334E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.63334E+01 0.00069  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.10538E+01 0.00096  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.91168E-02 0.00425  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000365 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00036E+04 0.00199 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00036E+04 0.00199 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.43337E+03 ;
RUNNING_TIME              (idx, 1)        =  1.88985E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.31883E-01  2.20000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.86423E+02  1.90100E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.10183E+00  9.74167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.96400E-01  1.76000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.88983E+02  2.44796E+02 ];
CPU_USAGE                 (idx, 1)        = 7.58461 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66726E+00 0.00027 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92320E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 188 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.87970E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73833E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.81779E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92226E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32841E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92618E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57449E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.04229E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75111E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.93456E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.02995E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44725E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.71305E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.36655E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37507E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33456E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55651E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.68403E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.29659E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23004E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01886E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.39593E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98214E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.30044E+16 0.00115  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43619E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -4.76913E-02 -1.05305E+28  2.31337E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.99628E-01 0.00224 ];
TH232_FISS                (idx, [1:   4]) = [  1.38946E+18 0.01163  1.48549E-02 0.01150 ];
U233_FISS                 (idx, [1:   4]) = [  8.04747E+19 0.00173  8.60457E-01 0.00062 ];
U235_FISS                 (idx, [1:   4]) = [  7.24055E+18 0.00578  7.74221E-02 0.00564 ];
U238_FISS                 (idx, [1:   4]) = [  2.33215E+14 1.00000  2.48942E-06 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  3.67387E+17 0.02077  3.92870E-03 0.02080 ];
PU240_FISS                (idx, [1:   4]) = [  2.34584E+16 0.09596  2.50948E-04 0.09598 ];
PU241_FISS                (idx, [1:   4]) = [  4.36968E+16 0.07966  4.67765E-04 0.08021 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21361E+19 0.00178  6.73607E-01 0.00100 ];
U233_CAPT                 (idx, [1:   4]) = [  1.19496E+19 0.00403  8.73593E-02 0.00363 ];
U235_CAPT                 (idx, [1:   4]) = [  2.59504E+18 0.01020  1.89751E-02 0.01025 ];
U238_CAPT                 (idx, [1:   4]) = [  1.38280E+15 0.39785  1.00834E-05 0.39788 ];
PU239_CAPT                (idx, [1:   4]) = [  1.84365E+17 0.03026  1.34854E-03 0.03037 ];
PU240_CAPT                (idx, [1:   4]) = [  8.11052E+16 0.05136  5.92109E-04 0.05090 ];
PU241_CAPT                (idx, [1:   4]) = [  1.21772E+16 0.14032  8.92509E-05 0.14006 ];
SM149_CAPT                (idx, [1:   4]) = [  1.86742E+17 0.03581  1.36586E-03 0.03600 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000365 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15762E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000365 1.00116E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 594104 5.94581E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 406248 4.06563E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 13 1.29844E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000365 1.00116E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 8.14907E-10 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36143E+20 1.2E-05  2.36143E+20 1.2E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36256E+19 2.5E-07  9.36256E+19 2.5E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36526E+20 0.00074  1.26934E+20 0.00056  9.59184E+18 0.00629 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30152E+20 0.00044  2.20560E+20 0.00032  9.59184E+18 0.00629 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30044E+20 0.00115  2.30044E+20 0.00115  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.26170E+22 0.00078  3.15005E+22 0.00072  1.11655E+21 0.00640 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.99100E+15 0.32285 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30155E+20 0.00044 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06701E+22 0.00096 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.38046E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.38046E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.55662E-01 0.17053 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.55492E-01 0.03034 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.97104E-04 0.06861 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.80088E+03 0.03714 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99987E-01 4.2E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.07208E-01 0.15354 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.07204E-01 0.15354 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52220E+00 1.1E-05 ];
FISSE                     (idx, [1:   2]) = [  1.99994E+02 2.5E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02513E+00 0.00134  1.02213E+00 0.00131  3.24695E-03 0.02877 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02716E+00 0.00044 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02664E+00 0.00114 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02716E+00 0.00044 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02718E+00 0.00043 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03531E+00 0.00059 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03485E+00 0.00046 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32151E-02 0.00420 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32166E-02 0.00324 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.74316E-01 0.00401 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.75431E-01 0.00132 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39270E-03 0.01804  2.35480E-04 0.06521  7.72401E-04 0.03604  5.97617E-04 0.04225  1.40125E-03 0.02524  3.19743E-04 0.05737  6.62121E-05 0.12707 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.02674E-01 0.04722  1.14799E-02 0.02964  3.22175E-02 0.00046  1.06896E-01 0.00208  3.01039E-01 0.00102  1.21141E+00 0.02067  3.24926E+00 0.12356 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35425E-03 0.02460  1.96109E-04 0.09460  8.01738E-04 0.05376  5.69516E-04 0.05898  1.42157E-03 0.03768  3.01043E-04 0.08117  6.42783E-05 0.18997 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.84065E-01 0.05688  1.24772E-02 0.00011  3.22092E-02 0.00055  1.06964E-01 0.00266  3.00962E-01 0.00137  1.26080E+00 0.00276  6.72132E+00 0.06515 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.27175E-07 0.00428  8.26761E-07 0.00430  9.59850E-07 0.04486 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.47922E-07 0.00438  8.47497E-07 0.00440  9.84189E-07 0.04484 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.15215E-03 0.02854  1.89247E-04 0.11149  7.55077E-04 0.05788  5.50210E-04 0.07043  1.30183E-03 0.04155  2.99973E-04 0.07971  5.58205E-05 0.20813 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.78586E-01 0.06486  1.24765E-02 0.00019  3.22141E-02 0.00075  1.06853E-01 0.00305  3.00646E-01 0.00172  1.26353E+00 0.00394  6.54549E+00 0.10004 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.32885E-07 0.03507  7.32645E-07 0.03507  6.29793E-07 0.13071 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.50805E-07 0.03493  7.50560E-07 0.03493  6.45530E-07 0.13085 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09553E-03 0.10605  1.15874E-04 0.39667  8.89912E-04 0.19612  8.48030E-04 0.22628  8.75716E-04 0.18755  3.08202E-04 0.30270  5.77992E-05 0.82126 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.17254E-01 0.20258  1.24823E-02 0.00014  3.22637E-02 0.00226  1.06751E-01 0.00633  3.01374E-01 0.00572  1.24921E+00 0.00855  5.96319E+00 0.44828 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.96143E-03 0.10743  1.08940E-04 0.37415  8.60238E-04 0.20197  7.93703E-04 0.23048  8.58409E-04 0.18086  2.93979E-04 0.31401  4.61591E-05 0.82097 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.06922E-01 0.18223  1.24823E-02 0.00014  3.22610E-02 0.00227  1.06767E-01 0.00633  3.01234E-01 0.00564  1.24863E+00 0.00847  5.96319E+00 0.44828 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.28589E+03 0.10138 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.18825E-07 0.00196 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.39287E-07 0.00170 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24604E-03 0.01439 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.96532E+03 0.01445 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.06937E-09 0.00251 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.01226E-05 0.04124  6.01631E-05 0.04128  7.02194E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.47795E-05 0.08677  5.47529E-05 0.08680  7.38300E-07 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.06209E-04 0.06965  2.05899E-04 0.06948  2.68589E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.52602E+01 0.03600 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.63334E+01 0.00069  4.05331E+01 0.00111 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 02:01:05 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02165E+00  9.97984E-01  9.90229E-01  1.00357E+00  1.00213E+00  9.96895E-01  9.90171E-01  9.97367E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.21299E-02 0.00088  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77870E-01 2.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.21638E-01 0.00053  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.36360E-01 0.00050  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66230E+00 0.00086  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.63067E+01 0.00071  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.63067E+01 0.00071  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.13659E+01 0.00094  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.01992E-02 0.00590  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000230 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00023E+04 0.00168 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00023E+04 0.00168 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.57931E+03 ;
RUNNING_TIME              (idx, 1)        =  2.08161E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.54183E-01  2.23000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.05414E+02  1.89918E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.19947E+00  9.76333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.14100E-01  1.77000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.08159E+02  2.46419E+02 ];
CPU_USAGE                 (idx, 1)        = 7.58697 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66656E+00 0.00026 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92615E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.88198E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73364E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.83769E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89809E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31996E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95254E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57052E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.62280E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38879E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.44902E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14233E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69298E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23763E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58312E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34658E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31172E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52599E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.72111E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.60591E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20981E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99657E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.14591E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.95560E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.30781E+16 0.00102  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28692E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.01329E-01 -2.23740E+28  2.43180E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.94490E-01 0.00191 ];
TH232_FISS                (idx, [1:   4]) = [  1.35458E+18 0.01113  1.44484E-02 0.01067 ];
U233_FISS                 (idx, [1:   4]) = [  7.94111E+19 0.00167  8.47320E-01 0.00063 ];
U235_FISS                 (idx, [1:   4]) = [  7.45390E+18 0.00489  7.95426E-02 0.00487 ];
PU239_FISS                (idx, [1:   4]) = [  6.82710E+17 0.01774  7.28729E-03 0.01788 ];
PU240_FISS                (idx, [1:   4]) = [  8.27961E+16 0.05033  8.83648E-04 0.05035 ];
PU241_FISS                (idx, [1:   4]) = [  1.52585E+17 0.03904  1.62793E-03 0.03897 ];
TH232_CAPT                (idx, [1:   4]) = [  9.05479E+19 0.00149  6.59426E-01 0.00092 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17898E+19 0.00428  8.58568E-02 0.00402 ];
U235_CAPT                 (idx, [1:   4]) = [  2.66523E+18 0.00931  1.94110E-02 0.00930 ];
U238_CAPT                 (idx, [1:   4]) = [  3.26221E+15 0.27040  2.36429E-05 0.26956 ];
PU239_CAPT                (idx, [1:   4]) = [  3.23620E+17 0.02659  2.35683E-03 0.02666 ];
PU240_CAPT                (idx, [1:   4]) = [  2.22123E+17 0.03072  1.61803E-03 0.03086 ];
PU241_CAPT                (idx, [1:   4]) = [  2.93218E+16 0.09024  2.13432E-04 0.09006 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90512E+17 0.03341  1.38750E-03 0.03346 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000230 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11691E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000230 1.00112E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 594468 5.95001E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 405745 4.06099E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 17 1.69870E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000230 1.00112E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -6.28643E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36455E+20 1.3E-05  2.36455E+20 1.3E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35835E+19 3.0E-07  9.35835E+19 3.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37342E+20 0.00071  1.27524E+20 0.00053  9.81779E+18 0.00686 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30925E+20 0.00042  2.21107E+20 0.00030  9.81779E+18 0.00686 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30781E+20 0.00102  2.30781E+20 0.00102  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.28777E+22 0.00065  3.17295E+22 0.00060  1.14821E+21 0.00628 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.91734E+15 0.27695 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30929E+20 0.00042 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06976E+22 0.00084 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29941E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29941E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  3.71520E-01 0.18726 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.51406E-01 0.03483 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.14029E-04 0.07581 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.26782E+03 0.06298 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99983E-01 4.7E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.56556E-01 0.17409 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.56548E-01 0.17409 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52668E+00 1.3E-05 ];
FISSE                     (idx, [1:   2]) = [  2.00084E+02 3.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02626E+00 0.00112  1.02261E+00 0.00114  3.47420E-03 0.02652 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02506E+00 0.00042 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02469E+00 0.00101 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02506E+00 0.00042 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02508E+00 0.00042 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00550E+00 0.00063 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00816E+00 0.00040 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36165E-02 0.00441 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35724E-02 0.00279 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81446E-01 0.00456 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81511E-01 0.00141 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41816E-03 0.01645  2.57924E-04 0.05899  7.89078E-04 0.03657  6.13883E-04 0.03849  1.35527E-03 0.02488  3.47190E-04 0.04858  5.48186E-05 0.13783 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.78990E-01 0.03775  1.18550E-02 0.02306  3.22036E-02 0.00056  1.06655E-01 0.00185  3.00921E-01 0.00098  1.23418E+00 0.01467  2.94899E+00 0.13317 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35550E-03 0.02311  2.51450E-04 0.08970  7.83977E-04 0.04817  5.90903E-04 0.05485  1.32275E-03 0.03616  3.62756E-04 0.06795  4.36697E-05 0.18481 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.64854E-01 0.04472  1.24777E-02 0.00014  3.21965E-02 0.00080  1.06724E-01 0.00224  3.01356E-01 0.00148  1.26269E+00 0.00362  7.20158E+00 0.06006 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.18125E-07 0.00395  8.17638E-07 0.00393  9.47137E-07 0.04390 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39437E-07 0.00358  8.38938E-07 0.00356  9.71357E-07 0.04381 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.38755E-03 0.02711  2.13542E-04 0.10871  8.08962E-04 0.05887  6.26267E-04 0.06745  1.33084E-03 0.04428  3.55782E-04 0.08238  5.21529E-05 0.21453 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.84191E-01 0.07014  1.24900E-02 0.00119  3.22090E-02 0.00114  1.06594E-01 0.00294  3.00804E-01 0.00172  1.26157E+00 0.00452  6.62624E+00 0.10642 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.23339E-07 0.03523  7.23392E-07 0.03524  5.76250E-07 0.12907 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.42628E-07 0.03523  7.42687E-07 0.03523  5.90575E-07 0.12826 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.76948E-03 0.12249  2.98989E-04 0.45101  1.03479E-03 0.23428  7.83382E-04 0.25962  1.24162E-03 0.25202  2.08479E-04 0.35048  2.02216E-04 0.61006 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.66735E-01 0.28992  1.24808E-02 9.0E-05  3.21950E-02 0.00224  1.06044E-01 0.00733  3.02673E-01 0.00631  1.24678E+00 0.03168  7.38032E+00 0.28390 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.67011E-03 0.11736  3.06643E-04 0.42311  1.03174E-03 0.23327  7.48175E-04 0.25144  1.17817E-03 0.24099  1.97027E-04 0.33938  2.08351E-04 0.60323 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.76925E-01 0.29564  1.24808E-02 9.0E-05  3.21953E-02 0.00225  1.06034E-01 0.00732  3.02725E-01 0.00626  1.24678E+00 0.03168  7.38070E+00 0.28392 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -5.21546E+03 0.11863 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.13835E-07 0.00237 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.35075E-07 0.00195 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32957E-03 0.01554 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.08866E+03 0.01492 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.08603E-09 0.00274 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  5.79773E-05 0.04838  5.79773E-05 0.04838  0.00000E+00 0.0E+00 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.51349E-05 0.09690  5.51349E-05 0.09690  0.00000E+00 0.0E+00 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.21914E-04 0.07370  2.22673E-04 0.07371  0.00000E+00 0.0E+00 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61777E+01 0.03362 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.63067E+01 0.00071  4.02402E+01 0.00101 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 02:20:22 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02067E+00  1.00170E+00  1.00525E+00  9.96090E-01  9.85502E-01  1.00242E+00  9.91595E-01  9.96772E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.22273E-02 0.00091  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77773E-01 2.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.20228E-01 0.00054  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.35045E-01 0.00051  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66863E+00 0.00094  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.63007E+01 0.00067  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.63007E+01 0.00067  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.18938E+01 0.00084  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.09150E-02 0.00518  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 999977 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  9.99977E+03 0.00168 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  9.99977E+03 0.00168 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.72600E+03 ;
RUNNING_TIME              (idx, 1)        =  2.27438E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.77717E-01  2.35333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.24496E+02  1.90820E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.30185E+00  1.02383E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.32500E-01  1.84000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.27437E+02  2.46555E+02 ];
CPU_USAGE                 (idx, 1)        = 7.58889 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66940E+00 0.00019 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92842E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.88283E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73123E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.77047E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89457E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31956E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95683E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56809E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.71847E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55866E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.53744E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16211E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76246E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.38712E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.92246E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33612E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30387E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51514E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.72821E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00615E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20284E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.98203E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.33986E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.95026E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.30849E+16 0.00097  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93022E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.56936E-01 -3.46524E+28  2.55459E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.95651E-01 0.00208 ];
TH232_FISS                (idx, [1:   4]) = [  1.34382E+18 0.01523  1.43472E-02 0.01493 ];
U233_FISS                 (idx, [1:   4]) = [  7.91818E+19 0.00165  8.45669E-01 0.00073 ];
U235_FISS                 (idx, [1:   4]) = [  7.42503E+18 0.00537  7.93096E-02 0.00539 ];
PU239_FISS                (idx, [1:   4]) = [  7.22159E+17 0.01701  7.70972E-03 0.01670 ];
PU240_FISS                (idx, [1:   4]) = [  9.72148E+16 0.04779  1.03794E-03 0.04761 ];
PU241_FISS                (idx, [1:   4]) = [  1.88816E+17 0.03594  2.01615E-03 0.03595 ];
TH232_CAPT                (idx, [1:   4]) = [  9.03667E+19 0.00161  6.57364E-01 0.00104 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17435E+19 0.00433  8.54238E-02 0.00408 ];
U235_CAPT                 (idx, [1:   4]) = [  2.65864E+18 0.00991  1.93401E-02 0.00984 ];
U238_CAPT                 (idx, [1:   4]) = [  3.46367E+15 0.27451  2.51370E-05 0.27360 ];
PU239_CAPT                (idx, [1:   4]) = [  3.66642E+17 0.02595  2.66758E-03 0.02601 ];
PU240_CAPT                (idx, [1:   4]) = [  2.80291E+17 0.03031  2.03851E-03 0.03019 ];
PU241_CAPT                (idx, [1:   4]) = [  4.15943E+16 0.07340  3.02692E-04 0.07356 ];
SM149_CAPT                (idx, [1:   4]) = [  1.93504E+17 0.03692  1.40736E-03 0.03689 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 999977 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11048E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 999977 1.00111E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 594810 5.95493E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 405151 4.05601E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 16 1.60733E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 999977 1.00111E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.91624E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36519E+20 1.4E-05  2.36519E+20 1.4E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35772E+19 3.3E-07  9.35772E+19 3.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37529E+20 0.00079  1.27574E+20 0.00053  9.95501E+18 0.00689 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31106E+20 0.00047  2.21151E+20 0.00031  9.95501E+18 0.00689 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30849E+20 0.00097  2.30849E+20 0.00097  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.30610E+22 0.00074  3.19045E+22 0.00067  1.15649E+21 0.00690 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.69720E+15 0.26204 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.31110E+20 0.00047 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06993E+22 0.00096 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.22606E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.22606E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.92404E-01 0.17959 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.12157E-01 0.03888 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.06933E-04 0.07191 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  3.80757E+03 0.03684 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 4.2E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.87158E-01 0.16118 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.87152E-01 0.16118 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52753E+00 1.3E-05 ];
FISSE                     (idx, [1:   2]) = [  2.00097E+02 3.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02523E+00 0.00117  1.02168E+00 0.00118  3.49304E-03 0.02588 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02455E+00 0.00047 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02466E+00 0.00096 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02455E+00 0.00047 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02457E+00 0.00047 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00583E+00 0.00060 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00923E+00 0.00044 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36108E-02 0.00425 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35590E-02 0.00312 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81933E-01 0.00479 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.82157E-01 0.00146 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.58703E-03 0.01710  2.51996E-04 0.05712  8.47657E-04 0.03304  6.28103E-04 0.04007  1.46055E-03 0.02778  3.24482E-04 0.05555  7.42442E-05 0.09746 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.22566E-01 0.04269  1.16005E-02 0.02757  3.21851E-02 0.00046  1.06728E-01 0.00167  3.01053E-01 0.00092  1.21605E+00 0.01809  4.55211E+00 0.09575 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47763E-03 0.02359  2.65595E-04 0.09084  8.51445E-04 0.04817  6.35605E-04 0.05158  1.36883E-03 0.03679  2.88112E-04 0.06799  6.80471E-05 0.13344 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.99837E-01 0.05811  1.24745E-02 0.00013  3.21725E-02 0.00072  1.06613E-01 0.00194  3.01353E-01 0.00131  1.25555E+00 0.00445  7.81342E+00 0.04732 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.20094E-07 0.00334  8.19946E-07 0.00334  8.84670E-07 0.04495 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.40704E-07 0.00325  8.40551E-07 0.00325  9.06905E-07 0.04485 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40831E-03 0.02663  2.27823E-04 0.10474  7.85171E-04 0.06017  6.03478E-04 0.06473  1.39808E-03 0.04510  3.30688E-04 0.08636  6.30630E-05 0.19096 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.13932E-01 0.07812  1.24731E-02 0.00021  3.22113E-02 0.00102  1.06394E-01 0.00276  3.00637E-01 0.00169  1.25619E+00 0.00588  7.44882E+00 0.08860 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.38998E-07 0.03625  7.38703E-07 0.03624  6.59054E-07 0.13699 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.58180E-07 0.03619  7.57870E-07 0.03618  6.77565E-07 0.13757 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.66720E-03 0.11057  1.73469E-04 0.45915  1.08128E-03 0.18332  5.99203E-04 0.25956  1.52930E-03 0.18369  2.25439E-04 0.32894  5.85018E-05 0.73873 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.84002E-01 0.35345  1.24794E-02 0.0E+00  3.21918E-02 0.00187  1.05673E-01 0.00394  3.01603E-01 0.00516  1.26403E+00 0.01047  7.90779E+00 0.29198 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.62748E-03 0.10913  1.60393E-04 0.48795  1.01520E-03 0.18195  5.89530E-04 0.24915  1.55599E-03 0.18185  2.37252E-04 0.32644  6.91089E-05 0.72486 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.99330E-01 0.35601  1.24794E-02 5.6E-09  3.21902E-02 0.00188  1.05673E-01 0.00394  3.01381E-01 0.00513  1.26403E+00 0.01047  7.90779E+00 0.29198 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -5.05518E+03 0.10491 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.17663E-07 0.00198 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.38184E-07 0.00162 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.51133E-03 0.01670 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.29363E+03 0.01647 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.09015E-09 0.00259 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  5.96149E-05 0.04864  5.96179E-05 0.04864  7.09554E-07 1.00000 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.40109E-05 0.07911  5.35779E-05 0.07980  2.02750E-06 1.00000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.13947E-04 0.07103  2.13718E-04 0.07093  2.58740E-04 1.00000 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58522E+01 0.03211 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.63007E+01 0.00067  4.02073E+01 0.00107 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 16:00:34' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 20])  = 'burn_benchmark_prova' ;
WORKING_DIRECTORY         (idx, [1: 21])  = '/home/cfx/tiago/vazao' ;
HOSTNAME                  (idx, [1:  9])  = 'cahill-lx' ;
CPU_TYPE                  (idx, [1: 47])  = 'Intel(R) Xeon(R) CPU           E5520  @ 2.27GHz' ;
CPU_MHZ                   (idx, 1)        = 29.0 ;
START_DATE                (idx, [1: 24])  = 'Sun Jan 30 22:32:55 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Mon Jan 31 02:35:35 2022' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1643592775766 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 8 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:   8]) = [  1.02193E+00  9.91637E-01  1.00328E+00  9.91986E-01  1.00247E+00  9.97641E-01  9.92966E-01  9.98099E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 30])  = '/opt/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 26])  = '/opt/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.23380E-02 0.00089  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77662E-01 2.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.18671E-01 0.00043  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.33593E-01 0.00040  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69223E+00 0.00096  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.62828E+01 0.00075  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.62827E+01 0.00075  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.24590E+01 0.00090  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.18613E-02 0.00525  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1000296 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00030E+04 0.00200 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00030E+04 0.00200 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.84147E+03 ;
RUNNING_TIME              (idx, 1)        =  2.42659E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  8.60367E-01  8.60367E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.00583E-01  2.28667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.39515E+02  1.50186E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.40540E+00  1.03550E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.49967E-01  1.74667E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.42657E+02  2.42657E+02 ];
CPU_USAGE                 (idx, 1)        = 7.58873 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  7.66455E+00 0.00028 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92885E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 48133.23 ;
ALLOC_MEMSIZE             (idx, 1)        = 7266.22;
MEMSIZE                   (idx, 1)        = 7165.40;
XS_MEMSIZE                (idx, 1)        = 6911.10;
MAT_MEMSIZE               (idx, 1)        = 160.21;
RES_MEMSIZE               (idx, 1)        = 11.82;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 82.27;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 100.81;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.88447E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73110E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.03718E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89302E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31908E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95992E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56791E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.73737E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60902E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.55406E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16678E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78310E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43240E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.02039E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33450E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30250E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51342E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.72764E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13069E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20155E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97991E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.39399E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94883E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  2.31047E+16 0.00102  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57353E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.12823E-01 -4.69927E+28  2.67799E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00081E+00 0.00215 ];
TH232_FISS                (idx, [1:   4]) = [  1.34286E+18 0.01574  1.43694E-02 0.01548 ];
U233_FISS                 (idx, [1:   4]) = [  7.89965E+19 0.00156  8.45508E-01 0.00064 ];
U235_FISS                 (idx, [1:   4]) = [  7.40524E+18 0.00548  7.92446E-02 0.00493 ];
PU239_FISS                (idx, [1:   4]) = [  7.40907E+17 0.01688  7.92879E-03 0.01674 ];
PU240_FISS                (idx, [1:   4]) = [  9.62228E+16 0.04576  1.02956E-03 0.04570 ];
PU241_FISS                (idx, [1:   4]) = [  1.87726E+17 0.03490  2.00865E-03 0.03490 ];
TH232_CAPT                (idx, [1:   4]) = [  9.06739E+19 0.00174  6.57718E-01 0.00090 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17363E+19 0.00361  8.51379E-02 0.00353 ];
U235_CAPT                 (idx, [1:   4]) = [  2.65298E+18 0.00945  1.92444E-02 0.00937 ];
U238_CAPT                 (idx, [1:   4]) = [  3.27542E+15 0.28911  2.36712E-05 0.28807 ];
PU239_CAPT                (idx, [1:   4]) = [  3.68635E+17 0.02582  2.67231E-03 0.02558 ];
PU240_CAPT                (idx, [1:   4]) = [  2.89953E+17 0.02725  2.10455E-03 0.02747 ];
PU241_CAPT                (idx, [1:   4]) = [  3.86211E+16 0.08831  2.80066E-04 0.08813 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80889E+17 0.03488  1.31288E-03 0.03496 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 1000296 1.00000E+06 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08205E+03 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 1000296 1.00108E+06 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 596207 5.96671E+05 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 404073 4.04395E+05 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 16 1.59798E+01 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 1000296 1.00108E+06 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.95812E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 4.0E-09  3.52494E-02 4.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36525E+20 1.5E-05  2.36525E+20 1.5E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35759E+19 3.3E-07  9.35759E+19 3.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37478E+20 0.00083  1.27584E+20 0.00058  9.89371E+18 0.00638 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31054E+20 0.00049  2.21160E+20 0.00034  9.89371E+18 0.00638 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.31047E+20 0.00102  2.31047E+20 0.00102  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32830E+22 0.00070  3.21298E+22 0.00065  1.15313E+21 0.00645 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.68740E+15 0.26249 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.31057E+20 0.00049 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07045E+22 0.00090 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.15837E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.15837E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.21322E-01 0.18683 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.82665E-01 0.03968 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.25114E-04 0.06730 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.00859E+03 0.01950 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 4.2E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.76849E-01 0.16527 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.76844E-01 0.16527 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52763E+00 1.5E-05 ];
FISSE                     (idx, [1:   2]) = [  2.00100E+02 3.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02335E+00 0.00138  1.01875E+00 0.00132  3.46253E-03 0.02844 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02478E+00 0.00050 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02382E+00 0.00102 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02478E+00 0.00050 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02480E+00 0.00049 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01053E+00 0.00065 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00982E+00 0.00038 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35490E-02 0.00455 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35494E-02 0.00267 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81596E-01 0.00552 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81540E-01 0.00153 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.46983E-03 0.01671  2.47419E-04 0.05895  8.02935E-04 0.03225  6.43931E-04 0.03596  1.38269E-03 0.02681  3.23441E-04 0.05686  6.94078E-05 0.11567 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.07780E-01 0.04707  1.17275E-02 0.02539  3.21849E-02 0.00052  1.06738E-01 0.00211  3.01255E-01 0.00097  1.19224E+00 0.02328  3.65054E+00 0.11048 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35894E-03 0.02192  2.10888E-04 0.08035  7.89442E-04 0.04293  6.37374E-04 0.05599  1.35377E-03 0.03807  2.89237E-04 0.07985  7.82343E-05 0.16594 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.24197E-01 0.07320  1.24773E-02 0.00010  3.21918E-02 0.00064  1.06824E-01 0.00251  3.01264E-01 0.00124  1.25538E+00 0.00373  7.00292E+00 0.05620 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.30860E-07 0.00322  8.30452E-07 0.00323  9.76643E-07 0.04990 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.50118E-07 0.00296  8.49700E-07 0.00298  9.99187E-07 0.04986 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.35473E-03 0.02829  2.17950E-04 0.09787  7.40806E-04 0.05763  6.65876E-04 0.06395  1.39086E-03 0.04533  2.82241E-04 0.09676  5.69978E-05 0.22916 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.88880E-01 0.08180  1.24800E-02 4.1E-05  3.21815E-02 0.00098  1.06722E-01 0.00288  3.01395E-01 0.00193  1.26255E+00 0.00520  7.77584E+00 0.07895 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.46791E-07 0.03521  7.45813E-07 0.03517  6.33226E-07 0.15023 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.64049E-07 0.03521  7.63058E-07 0.03517  6.48730E-07 0.14955 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.77893E-03 0.12591  4.98829E-05 0.50436  4.11600E-04 0.26533  5.75058E-04 0.25736  1.42946E-03 0.16706  1.42973E-04 0.50505  1.69961E-04 0.61355 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.47962E-01 0.33440  1.24799E-02 4.0E-05  3.21279E-02 0.00290  1.06304E-01 0.00667  3.01136E-01 0.00488  1.24911E+00 0.01474  9.82652E+00 0.04037 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.82308E-03 0.12473  5.38646E-05 0.50212  4.14662E-04 0.25206  5.98977E-04 0.24979  1.43709E-03 0.17055  1.31432E-04 0.47294  1.87058E-04 0.60898 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  5.35689E-01 0.33165  1.24799E-02 4.0E-05  3.21282E-02 0.00286  1.06301E-01 0.00668  3.00932E-01 0.00473  1.24911E+00 0.01474  9.82652E+00 0.04037 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65387E+03 0.12270 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.25629E-07 0.00207 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.44778E-07 0.00174 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.31066E-03 0.01479 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.01149E+03 0.01504 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.12155E-09 0.00238 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.16564E-05 0.03938  6.10835E-05 0.04105  1.15803E-06 0.70500 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.95835E-05 0.08977  5.83483E-05 0.09204  2.31568E-06 0.70741 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.33073E-04 0.06633  2.31884E-04 0.06678  5.96896E-04 0.71236 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60420E+01 0.03792 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.62827E+01 0.00075  4.02178E+01 0.00114 ];

