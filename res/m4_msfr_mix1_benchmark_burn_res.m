
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 05:32:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04532E+00  1.00433E+00  1.00683E+00  9.99682E-01  9.94961E-01  1.00216E+00  9.98539E-01  9.99414E-01  9.97201E-01  9.95392E-01  9.74900E-01  9.93838E-01  9.96822E-01  9.92891E-01  9.99754E-01  9.97455E-01  9.97233E-01  1.00327E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.99973E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70003E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39684E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53558E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61790E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46724E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45814E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.15131E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25469E+00 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500600 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50067E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50067E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.86359E+02 ;
RUNNING_TIME              (idx, 1)        =  2.06287E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.99333E-02  9.99333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.95256E+01  1.95256E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.22045E+00  1.16920E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.95053E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.03397 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97794E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.95302E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  2.80441E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.20544E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13402E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.80441E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.20544E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.69538E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.43081E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.69538E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.43081E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29682E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.13304E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.31582E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.45677E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.66381E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.52010E+18 0.00413  1.62137E-02 0.00408 ];
U233_FISS                 (idx, [1:   4]) = [  9.22301E+19 0.00049  9.83786E-01 6.7E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.15766E+19 0.00052  7.54251E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34695E+19 0.00124  1.10941E-01 0.00121 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002699 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.17160E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002699 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5130816 5.13557E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3961916 3.96545E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909967 9.10695E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002699 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.41561E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36440E+20 3.6E-06  2.36440E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.2E-07  9.37766E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21430E+20 0.00030  1.08105E+20 0.00022  1.33250E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15207E+20 0.00017  2.01882E+20 0.00012  1.33250E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36419E+20 0.00035  2.36419E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.12686E+22 0.00025  2.96455E+22 0.00022  1.62308E+21 0.00194 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.15314E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.36738E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05515E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.23714E-01 0.07467 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.85980E-01 0.00830 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.94766E-04 0.01466 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.95433E+03 0.01979 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08940E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.2E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.95913E-01 0.06675 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.59843E-01 0.06675 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52131E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.99523E-01 0.00041  6.22851E-02 0.00039  2.02833E-04 0.00834 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.99902E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  1.00014E+00 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.99902E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.10006E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04681E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04535E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30574E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30740E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.46057E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46648E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.25083E-03 0.00597  2.67232E-04 0.01958  8.07770E-04 0.01120  5.88245E-04 0.01291  1.26738E-03 0.00943  2.71024E-04 0.01846  4.91784E-05 0.04662 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.41970E-01 0.01490  1.24757E-02 2.7E-05  3.23281E-02 8.7E-05  1.06089E-01 0.00058  2.97245E-01 0.00026  1.23608E+00 0.00020  4.46729E+00 0.04216 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27147E-03 0.00854  2.59744E-04 0.02835  8.17351E-04 0.01664  5.84803E-04 0.02060  1.29054E-03 0.01360  2.66864E-04 0.02885  5.21754E-05 0.06858 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46385E-01 0.02427  1.24755E-02 3.4E-05  3.23315E-02 0.00013  1.06120E-01 0.00086  2.97198E-01 0.00036  1.23648E+00 0.00026  6.15778E+00 0.02938 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.34427E-07 0.00119  8.34055E-07 0.00119  9.41848E-07 0.01769 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.33972E-07 0.00111  8.33601E-07 0.00112  9.41375E-07 0.01769 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23808E-03 0.00830  2.55740E-04 0.03209  8.13610E-04 0.01703  5.78544E-04 0.02017  1.27497E-03 0.01362  2.65542E-04 0.03233  4.96760E-05 0.07307 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.41857E-01 0.02401  1.24760E-02 4.2E-05  3.23356E-02 0.00016  1.06037E-01 0.00094  2.97216E-01 0.00042  1.23637E+00 0.00034  6.34016E+00 0.04052 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.40509E-07 0.01707  7.40408E-07 0.01707  7.72858E-07 0.05499 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.40289E-07 0.01707  7.40187E-07 0.01707  7.72898E-07 0.05512 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80646E-03 0.03982  2.47603E-04 0.14056  7.27392E-04 0.08049  4.44581E-04 0.10595  1.09161E-03 0.06273  2.40777E-04 0.12377  5.44936E-05 0.23429 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.02305E-01 0.10605  1.24771E-02 0.00011  3.23239E-02 0.00046  1.06258E-01 0.00343  2.96739E-01 0.00134  1.23496E+00 0.00109  6.75662E+00 0.10261 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81539E-03 0.03909  2.40799E-04 0.14065  7.17504E-04 0.07781  4.48904E-04 0.10634  1.10549E-03 0.06141  2.45278E-04 0.12040  5.74136E-05 0.24121 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.09492E-01 0.10822  1.24771E-02 0.00011  3.23251E-02 0.00046  1.06319E-01 0.00348  2.96757E-01 0.00134  1.23496E+00 0.00108  6.75662E+00 0.10261 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.82607E+03 0.03674 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.29479E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.29027E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24010E-03 0.00547 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90685E+03 0.00551 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.23400E-09 0.00121 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.27390E-05 0.00447  7.27382E-05 0.00449  3.25151E-06 0.23747 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.89327E-05 0.01596  7.88711E-05 0.01606  4.99044E-06 0.34437 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.78724E-04 0.01398  4.78381E-04 0.01399  5.84342E-04 0.23369 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75471E+01 0.01132 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45814E+01 0.00027  4.04287E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 06:04:29 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01337E+00  9.99449E-01  9.99815E-01  1.00401E+00  9.99272E-01  9.98384E-01  9.95053E-01  1.00238E+00  9.93335E-01  1.00258E+00  1.00690E+00  9.89873E-01  1.00244E+00  9.95876E-01  9.96268E-01  1.00086E+00  9.98717E-01  1.00142E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.99204E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70080E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39875E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53713E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61654E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47566E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46659E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16121E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25245E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500119 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50051E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50051E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.76695E+02 ;
RUNNING_TIME              (idx, 1)        =  5.22753E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.84100E-01  8.41667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.10679E+01  3.15423E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.52333E-02  1.52333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.03182E+00  2.81132E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.95121E+01  3.04636E+02 ];
CPU_USAGE                 (idx, 1)        = 9.11893 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98435E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.04117E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  5.76154E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60290E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65121E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.46622E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.03282E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26473E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46894E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.44490E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65236E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.38175E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.74202E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06314E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17816E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44278E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35678E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17535E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55430E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59384E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38169E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.38774E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51226E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50421E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.27554E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.55530E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08101E-04 -2.38684E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.77081E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.54704E+18 0.00386  1.64874E-02 0.00380 ];
U233_FISS                 (idx, [1:   4]) = [  9.22539E+19 0.00051  9.83222E-01 6.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.58697E+14 0.25366  3.83161E-06 0.25366 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26552E+19 0.00055  7.49380E-01 0.00025 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34555E+19 0.00141  1.08827E-01 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.18241E+14 0.44498  9.59295E-07 0.44499 ];
SM149_CAPT                (idx, [1:   4]) = [  1.26714E+16 0.04494  1.02565E-04 0.04498 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002054 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16423E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002054 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5170717 5.17589E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3924179 3.92785E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907158 9.07910E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002054 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.21541E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36434E+20 3.8E-06  2.36434E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37763E+19 1.3E-07  9.37763E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23590E+20 0.00030  1.10055E+20 0.00022  1.35353E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17366E+20 0.00017  2.03831E+20 0.00012  1.35353E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38882E+20 0.00037  2.38882E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.16239E+22 0.00026  2.99774E+22 0.00024  1.64646E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.16895E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39056E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06817E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51193E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51193E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.58833E-01 0.07479 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.81415E-01 0.00854 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.32499E-04 0.01633 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.81515E+03 0.01952 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09217E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.25004E-01 0.06261 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.86454E-01 0.06261 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52125E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.90363E-01 0.00044  6.16965E-02 0.00042  1.99107E-04 0.00839 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.90186E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.89804E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.90186E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08911E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04722E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04750E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30519E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30463E-02 0.00100 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47649E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.47345E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.27180E-03 0.00510  2.67454E-04 0.01912  7.87878E-04 0.01091  6.03909E-04 0.01254  1.29743E-03 0.00877  2.67864E-04 0.01912  4.72694E-05 0.04646 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.38167E-01 0.01443  1.24755E-02 2.6E-05  3.23351E-02 9.2E-05  1.06177E-01 0.00059  2.97340E-01 0.00027  1.23240E+00 0.00252  4.33630E+00 0.04391 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19701E-03 0.00800  2.69481E-04 0.02986  7.78081E-04 0.01558  5.88139E-04 0.01976  1.25061E-03 0.01261  2.61166E-04 0.02811  4.95360E-05 0.07454 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.43585E-01 0.02448  1.24754E-02 3.4E-05  3.23365E-02 0.00014  1.06211E-01 0.00087  2.97421E-01 0.00039  1.23564E+00 0.00030  6.35194E+00 0.02960 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.40862E-07 0.00117  8.40662E-07 0.00117  8.98639E-07 0.01544 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32702E-07 0.00110  8.32504E-07 0.00111  8.89955E-07 0.01542 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20797E-03 0.00873  2.67225E-04 0.02986  7.73118E-04 0.01758  5.88706E-04 0.02104  1.27572E-03 0.01444  2.59571E-04 0.03063  4.36226E-05 0.07876 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.30627E-01 0.02414  1.24762E-02 4.5E-05  3.23327E-02 0.00015  1.06190E-01 0.00101  2.97329E-01 0.00044  1.23501E+00 0.00041  6.14809E+00 0.04435 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.46797E-07 0.01739  7.46275E-07 0.01739  8.28454E-07 0.04911 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.39811E-07 0.01738  7.39293E-07 0.01739  8.20944E-07 0.04912 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89036E-03 0.04035  2.60122E-04 0.13136  6.75321E-04 0.07250  5.70279E-04 0.07983  1.11767E-03 0.06651  2.19967E-04 0.13188  4.69945E-05 0.30193 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.30226E-01 0.10025  1.24771E-02 0.00011  3.23162E-02 0.00041  1.05941E-01 0.00282  2.97060E-01 0.00148  1.23487E+00 0.00115  5.32187E+00 0.13406 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89446E-03 0.03997  2.67081E-04 0.12835  6.75458E-04 0.07255  5.71983E-04 0.07864  1.11655E-03 0.06583  2.13990E-04 0.12618  4.93991E-05 0.29247 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.32216E-01 0.09847  1.24771E-02 0.00010  3.23146E-02 0.00041  1.05967E-01 0.00284  2.97050E-01 0.00145  1.23486E+00 0.00115  5.30641E+00 0.13364 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.91368E+03 0.03722 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.35096E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.26984E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21362E-03 0.00605 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84967E+03 0.00615 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17804E-09 0.00130 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.18166E-05 0.00471  7.18067E-05 0.00475  2.75690E-06 0.25890 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.63068E-05 0.01533  7.62304E-05 0.01538  3.23888E-06 0.33814 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.16867E-04 0.01579  4.16734E-04 0.01587  4.50236E-04 0.25470 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73768E+01 0.01128 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46659E+01 0.00026  4.04652E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 06:38:04 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01664E+00  9.93662E-01  9.99103E-01  9.94204E-01  9.95942E-01  1.00127E+00  9.97457E-01  1.00352E+00  9.89279E-01  1.00449E+00  9.98378E-01  9.93316E-01  1.00153E+00  1.00564E+00  9.95132E-01  1.00601E+00  9.93166E-01  1.01127E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.97708E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70229E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39939E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53712E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61345E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48765E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47857E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18311E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24847E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500682 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50056E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50056E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.75830E+02 ;
RUNNING_TIME              (idx, 1)        =  8.58528E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.69433E-01  8.53333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  8.45377E+01  3.34698E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.68833E-02  1.16500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.00602E+00  3.97415E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.19253E+01  4.61597E+02 ];
CPU_USAGE                 (idx, 1)        = 9.03675 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98462E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.94774E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.37583E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65300E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.22796E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.84837E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29189E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49655E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49282E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.73857E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39460E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.84161E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.93890E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89675E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60071E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49515E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34556E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33302E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54243E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.67073E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17866E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.00046E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51024E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.31940E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.48767E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.68267E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.75933E-04 -1.27164E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.94597E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.57141E+18 0.00388  1.67437E-02 0.00384 ];
U233_FISS                 (idx, [1:   4]) = [  9.21501E+19 0.00052  9.81889E-01 6.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.02276E+16 0.04948  1.09052E-04 0.04954 ];
TH232_CAPT                (idx, [1:   4]) = [  9.41279E+19 0.00056  7.44079E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34183E+19 0.00138  1.06073E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  3.94595E+15 0.07933  3.12116E-05 0.07941 ];
SM149_CAPT                (idx, [1:   4]) = [  6.14456E+16 0.02003  4.85905E-04 0.02008 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002229 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15025E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002229 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5220847 5.22596E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3873637 3.87705E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907745 9.08494E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002229 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.73809E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36424E+20 3.7E-06  2.36424E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37757E+19 1.2E-07  9.37757E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26480E+20 0.00030  1.12723E+20 0.00020  1.37567E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20256E+20 0.00017  2.06499E+20 0.00011  1.37567E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42067E+20 0.00035  2.42067E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.20879E+22 0.00026  3.04186E+22 0.00023  1.66932E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.19922E+19 0.00134 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42248E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08529E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51432E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51432E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.55472E-01 0.05380 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07004E-01 0.01143 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.77581E-04 0.02059 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.57537E+03 0.02303 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09155E-01 0.00012 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.42270E-01 0.04109 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.83990E-01 0.04109 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52117E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77560E-01 0.00043  6.08940E-02 0.00042  1.96504E-04 0.00807 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.77103E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.76740E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.77103E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07479E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05418E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05321E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29616E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29721E-02 0.00100 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48060E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48315E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.32719E-03 0.00531  2.67372E-04 0.01905  8.15702E-04 0.01020  6.15560E-04 0.01265  1.30656E-03 0.00882  2.70890E-04 0.01913  5.11079E-05 0.04681 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.38938E-01 0.01526  1.24758E-02 2.9E-05  3.23364E-02 9.2E-05  1.05999E-01 0.00057  2.97475E-01 0.00028  1.23207E+00 0.00252  4.17309E+00 0.04356 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22057E-03 0.00793  2.56460E-04 0.02874  7.85634E-04 0.01712  6.06927E-04 0.02022  1.26661E-03 0.01330  2.54751E-04 0.02974  5.01913E-05 0.06632 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44537E-01 0.02276  1.24754E-02 4.2E-05  3.23336E-02 0.00012  1.06131E-01 0.00090  2.97546E-01 0.00041  1.23555E+00 0.00029  6.07546E+00 0.02985 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.54273E-07 0.00112  8.54006E-07 0.00112  9.37474E-07 0.01535 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.35039E-07 0.00103  8.34779E-07 0.00103  9.16292E-07 0.01531 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22371E-03 0.00840  2.62253E-04 0.03041  7.93706E-04 0.01758  5.98702E-04 0.02087  1.26718E-03 0.01501  2.54492E-04 0.03143  4.73734E-05 0.07886 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.32346E-01 0.02465  1.24749E-02 5.0E-05  3.23366E-02 0.00016  1.06020E-01 0.00096  2.97620E-01 0.00049  1.23541E+00 0.00041  6.21175E+00 0.04288 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.55377E-07 0.01711  7.55096E-07 0.01712  7.89641E-07 0.05038 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.38500E-07 0.01711  7.38224E-07 0.01712  7.72079E-07 0.05044 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.69545E-03 0.04159  2.57365E-04 0.13587  7.14726E-04 0.07530  4.35103E-04 0.08826  1.05931E-03 0.06366  1.95700E-04 0.13728  3.32459E-05 0.33724 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.81398E-01 0.06405  1.24756E-02 0.00014  3.23365E-02 0.00050  1.05853E-01 0.00292  2.97416E-01 0.00155  1.23454E+00 0.00137  6.55270E+00 0.13203 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.69164E-03 0.04080  2.53731E-04 0.13592  7.31066E-04 0.07397  4.27728E-04 0.08720  1.04985E-03 0.06134  1.94094E-04 0.13499  3.51711E-05 0.33431 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.81878E-01 0.06473  1.24756E-02 0.00014  3.23364E-02 0.00049  1.05870E-01 0.00291  2.97339E-01 0.00151  1.23453E+00 0.00137  6.55270E+00 0.13203 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.60197E+03 0.03877 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.49521E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.30396E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18141E-03 0.00493 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.74634E+03 0.00506 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99604E-09 0.00099 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.98626E-05 0.00575  6.98875E-05 0.00574  5.65332E-07 0.50687 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.71798E-05 0.02266  6.70960E-05 0.02267  9.93600E-07 0.63401 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.66356E-04 0.01971  2.66843E-04 0.01976  1.13238E-04 0.49976 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74176E+01 0.01137 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47857E+01 0.00027  4.06272E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 07:12:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01749E+00  1.00078E+00  1.00452E+00  9.99133E-01  9.94638E-01  9.97931E-01  9.95755E-01  9.98075E-01  1.00037E+00  9.97937E-01  1.00023E+00  1.00390E+00  1.00555E+00  9.92613E-01  9.98434E-01  9.93155E-01  1.00332E+00  9.96174E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.96566E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70343E-01 9.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40011E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53739E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60676E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49961E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49053E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20403E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24486E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500433 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50043E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50043E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.07935E+03 ;
RUNNING_TIME              (idx, 1)        =  1.19935E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.52717E-01  8.32833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.18508E+02  3.39707E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.02167E-02  1.33333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.20512E+01  4.04508E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.15936E+02  4.85167E+02 ];
CPU_USAGE                 (idx, 1)        = 8.99944 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98503E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.90524E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.49409E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66892E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.13172E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88706E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31800E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57572E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50575E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.79463E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.55659E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.99878E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.13763E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79477E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74281E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48829E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36206E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34570E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55948E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39393E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42575E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07432E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53916E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.12799E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.65111E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.75434E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12748E-03 -2.48945E+26  2.21046E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.08104E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.56930E+18 0.00380  1.67375E-02 0.00374 ];
U233_FISS                 (idx, [1:   4]) = [  9.19260E+19 0.00051  9.80492E-01 6.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.30601E+16 0.02443  4.59400E-04 0.02448 ];
TH232_CAPT                (idx, [1:   4]) = [  9.50365E+19 0.00053  7.41151E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33762E+19 0.00142  1.04316E-01 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.52826E+16 0.04090  1.19248E-04 0.04096 ];
SM149_CAPT                (idx, [1:   4]) = [  9.74141E+16 0.01510  7.59692E-04 0.01509 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001740 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14725E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001740 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5253052 5.25834E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3840982 3.84468E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907706 9.08448E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001740 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.03028E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36415E+20 3.7E-06  2.36415E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37750E+19 1.2E-07  9.37750E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28295E+20 0.00029  1.14404E+20 0.00021  1.38913E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22070E+20 0.00017  2.08179E+20 0.00012  1.38913E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43858E+20 0.00036  2.43858E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.23682E+22 0.00024  3.06828E+22 0.00022  1.68542E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.21545E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44225E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09623E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51441E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51441E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.96039E-01 0.05006 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.11504E-01 0.01159 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.27390E-04 0.02175 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.74621E+03 0.02137 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09158E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.90628E-01 0.03694 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.28011E-01 0.03694 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52109E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.69079E-01 0.00044  6.03881E-02 0.00042  1.93564E-04 0.00919 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.69157E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.69525E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.69157E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06596E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05871E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06030E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29023E-02 0.00125 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28801E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49359E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48875E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34188E-03 0.00576  2.66969E-04 0.01893  8.28909E-04 0.01103  6.11308E-04 0.01294  1.31481E-03 0.00858  2.68827E-04 0.01914  5.10541E-05 0.04814 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.41384E-01 0.01540  1.24760E-02 3.1E-05  3.23302E-02 9.2E-05  1.06037E-01 0.00061  2.97518E-01 0.00027  1.23549E+00 0.00025  4.08954E+00 0.04532 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24854E-03 0.00838  2.59636E-04 0.03060  8.03568E-04 0.01620  5.80946E-04 0.01868  1.29042E-03 0.01360  2.65442E-04 0.02947  4.85322E-05 0.06741 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44990E-01 0.02154  1.24766E-02 3.2E-05  3.23285E-02 0.00014  1.06115E-01 0.00085  2.97524E-01 0.00038  1.23540E+00 0.00031  6.37056E+00 0.02887 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.66294E-07 0.00124  8.65946E-07 0.00123  9.76727E-07 0.01741 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39443E-07 0.00116  8.39106E-07 0.00116  9.46356E-07 0.01740 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19080E-03 0.00931  2.62339E-04 0.03088  7.84351E-04 0.01812  5.76030E-04 0.02107  1.26066E-03 0.01430  2.59075E-04 0.03240  4.83441E-05 0.07692 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44513E-01 0.02533  1.24761E-02 4.9E-05  3.23320E-02 0.00017  1.06151E-01 0.00110  2.97438E-01 0.00046  1.23548E+00 0.00041  6.56382E+00 0.04163 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.65161E-07 0.01708  7.65030E-07 0.01708  7.60393E-07 0.05350 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.41587E-07 0.01706  7.41461E-07 0.01706  7.36765E-07 0.05350 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.79398E-03 0.04468  2.52297E-04 0.12871  6.79433E-04 0.08706  5.61048E-04 0.09019  1.03027E-03 0.06832  2.26990E-04 0.13231  4.39416E-05 0.28928 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.62148E-01 0.11202  1.24747E-02 0.00016  3.23141E-02 0.00048  1.06040E-01 0.00307  2.97575E-01 0.00171  1.23544E+00 0.00111  7.11884E+00 0.11760 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.77845E-03 0.04328  2.41597E-04 0.12639  6.76500E-04 0.08378  5.68018E-04 0.08738  1.02088E-03 0.06715  2.27639E-04 0.12850  4.38173E-05 0.27391 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.64391E-01 0.11205  1.24746E-02 0.00016  3.23134E-02 0.00047  1.06041E-01 0.00305  2.97551E-01 0.00170  1.23543E+00 0.00112  7.13532E+00 0.11716 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.67803E+03 0.04214 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60843E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.34161E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.15632E-03 0.00584 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.66698E+03 0.00585 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94449E-09 0.00087 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75637E-05 0.00809  6.75451E-05 0.00809  5.89314E-07 0.59112 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.41664E-05 0.02806  6.41609E-05 0.02806  5.98872E-07 0.59557 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.17515E-04 0.02128  2.17945E-04 0.02130  9.70524E-05 0.57713 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72213E+01 0.01165 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49053E+01 0.00025  4.07600E+01 0.00032 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 07:47:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01877E+00  1.00403E+00  9.98104E-01  9.98378E-01  9.93988E-01  9.99665E-01  1.00136E+00  1.00994E+00  9.90859E-01  9.98561E-01  9.99489E-01  9.95327E-01  9.89715E-01  1.00358E+00  1.00037E+00  9.96961E-01  1.00383E+00  9.97078E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.96181E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70382E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39768E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53489E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60780E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51218E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50307E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23594E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24673E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500660 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50056E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50056E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.39271E+03 ;
RUNNING_TIME              (idx, 1)        =  1.55265E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.38100E-01  8.53833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.53723E+02  3.52145E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.41000E-02  1.38833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.63850E+01  4.33373E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.50978E+02  4.91678E+02 ];
CPU_USAGE                 (idx, 1)        = 8.96987 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98522E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.87177E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.51966E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66337E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.51818E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90837E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33127E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57978E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49869E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.97249E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.67664E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.26822E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.23387E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69861E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85314E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93673E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32596E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31794E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52133E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34338E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81464E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07075E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51089E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.49982E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.69524E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.89447E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.28911E-03 -5.05428E+26  2.21303E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.32153E-01 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.59583E+18 0.00407  1.70147E-02 0.00405 ];
U233_FISS                 (idx, [1:   4]) = [  9.16206E+19 0.00051  9.76838E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.63437E+17 0.01249  1.74251E-03 0.01248 ];
TH232_CAPT                (idx, [1:   4]) = [  9.68137E+19 0.00055  7.37365E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33924E+19 0.00136  1.02004E-01 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  5.62994E+16 0.02134  4.28753E-04 0.02134 ];
SM149_CAPT                (idx, [1:   4]) = [  1.40565E+17 0.01306  1.07066E-03 0.01305 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002247 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14985E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002247 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5302797 5.30792E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3788395 3.79179E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911055 9.11792E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002247 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 6.23986E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36404E+20 3.5E-06  2.36404E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31307E+20 0.00031  1.17253E+20 0.00022  1.40543E+19 0.00187 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.25080E+20 0.00018  2.11026E+20 0.00012  1.40543E+19 0.00187 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47362E+20 0.00038  2.47362E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29020E+22 0.00026  3.12040E+22 0.00024  1.69802E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.25558E+19 0.00159 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47636E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11510E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51188E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51188E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.08306E-01 0.05066 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33501E-01 0.01183 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.95715E-04 0.02194 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.41562E+03 0.02284 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08823E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.44074E-01 0.03982 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.85348E-01 0.03982 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52103E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.55917E-01 0.00043  5.95517E-02 0.00043  1.92509E-04 0.00879 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55761E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.55755E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55761E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05166E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05785E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05848E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29141E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29037E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51283E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51459E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41598E-03 0.00552  2.72663E-04 0.01930  8.40876E-04 0.01086  6.27111E-04 0.01267  1.33673E-03 0.00867  2.84131E-04 0.01882  5.44644E-05 0.04430 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53056E-01 0.01564  1.24128E-02 0.00355  3.23293E-02 9.3E-05  1.06054E-01 0.00057  2.97431E-01 0.00026  1.23621E+00 0.00026  4.91903E+00 0.03874 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28830E-03 0.00787  2.64192E-04 0.02847  8.12670E-04 0.01595  6.04034E-04 0.01940  1.28521E-03 0.01269  2.74438E-04 0.02779  4.77622E-05 0.06796 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.43578E-01 0.02116  1.24745E-02 4.6E-05  3.23267E-02 0.00013  1.06032E-01 0.00077  2.97365E-01 0.00038  1.23661E+00 0.00034  6.58879E+00 0.02728 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80980E-07 0.00130  8.80601E-07 0.00130  1.00119E-06 0.01666 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.42075E-07 0.00120  8.41713E-07 0.00121  9.56791E-07 0.01659 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21209E-03 0.00902  2.58846E-04 0.03304  8.19298E-04 0.01820  5.79452E-04 0.02032  1.22814E-03 0.01456  2.75040E-04 0.03203  5.13170E-05 0.06863 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.46244E-01 0.02394  1.24751E-02 5.7E-05  3.23249E-02 0.00015  1.06159E-01 0.00108  2.97351E-01 0.00047  1.23675E+00 0.00040  6.13511E+00 0.04126 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.75953E-07 0.01705  7.75665E-07 0.01704  8.81515E-07 0.06441 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.41788E-07 0.01705  7.41515E-07 0.01704  8.42170E-07 0.06432 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91021E-03 0.04358  2.33561E-04 0.13758  7.26036E-04 0.08118  5.24042E-04 0.09472  1.12461E-03 0.06819  2.71954E-04 0.13983  3.00113E-05 0.32909 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.12139E-01 0.06972  1.24750E-02 0.00016  3.23193E-02 0.00049  1.05751E-01 0.00287  2.98192E-01 0.00174  1.23876E+00 0.00144  6.14486E+00 0.13882 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93083E-03 0.04286  2.35248E-04 0.13746  7.22382E-04 0.08186  5.30849E-04 0.09408  1.14077E-03 0.06733  2.71494E-04 0.13934  3.00870E-05 0.30825 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.12075E-01 0.06756  1.24749E-02 0.00016  3.23184E-02 0.00048  1.05757E-01 0.00286  2.98169E-01 0.00173  1.23880E+00 0.00143  6.14486E+00 0.13882 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.76552E+03 0.04041 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.72999E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.34449E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19564E-03 0.00611 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.66200E+03 0.00623 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92884E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84155E-05 0.00893  6.84332E-05 0.00893  9.41311E-07 0.42010 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.42336E-05 0.02563  6.42250E-05 0.02578  8.88983E-07 0.53195 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.88383E-04 0.02148  1.88426E-04 0.02155  1.86091E-04 0.40760 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71574E+01 0.01099 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50307E+01 0.00027  4.08236E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 08:23:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02208E+00  1.00006E+00  9.98505E-01  9.97943E-01  9.99642E-01  1.00046E+00  9.94951E-01  1.00231E+00  1.00313E+00  1.00035E+00  9.94762E-01  9.89196E-01  9.96571E-01  9.96780E-01  1.00038E+00  9.99250E-01  1.00123E+00  1.00240E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95732E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70427E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39515E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53222E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60528E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52275E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51363E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26491E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24814E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500720 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50038E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50038E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.70934E+03 ;
RUNNING_TIME              (idx, 1)        =  1.90928E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.24983E-01  8.68833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.89267E+02  3.55445E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.74167E-02  1.33167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.07242E+01  4.33920E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.86636E+02  5.04468E+02 ];
CPU_USAGE                 (idx, 1)        = 8.95281 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98528E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.85243E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.59071E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67344E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.26914E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94567E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35600E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61301E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50579E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09495E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.78829E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.68001E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.40437E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25482E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94756E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35205E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33667E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32586E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53206E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.01455E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17316E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11241E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54479E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.31915E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.80188E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.99474E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.43284E-03 -7.57961E+26  2.21555E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.51811E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.60134E+18 0.00395  1.70819E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  9.11967E+19 0.00052  9.72826E-01 9.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.36667E+17 0.00886  3.59136E-03 0.00885 ];
TH232_CAPT                (idx, [1:   4]) = [  9.80439E+19 0.00054  7.33800E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33251E+19 0.00139  9.97347E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  1.19679E+17 0.01430  8.95519E-04 0.01426 ];
SM149_CAPT                (idx, [1:   4]) = [  1.56085E+17 0.01284  1.16811E-03 0.01282 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001506 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14206E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001506 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5341842 5.34728E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3748119 3.75177E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911545 9.12371E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001506 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.34693E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36387E+20 3.6E-06  2.36387E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37700E+19 1.1E-07  9.37700E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33577E+20 0.00028  1.19391E+20 0.00021  1.41857E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27347E+20 0.00017  2.13161E+20 0.00012  1.41857E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49869E+20 0.00036  2.49869E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.33025E+22 0.00025  3.15891E+22 0.00023  1.71338E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.27983E+19 0.00153 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.50145E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12902E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50807E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50807E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.44960E-01 0.05588 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31668E-01 0.01278 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.80711E-04 0.02522 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.50216E+03 0.02254 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08767E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.01350E-01 0.04282 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.46447E-01 0.04282 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52093E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.45762E-01 0.00044  5.89252E-02 0.00043  1.89148E-04 0.00868 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.46101E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.46095E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.46101E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04109E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05775E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05900E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29154E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28969E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54090E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53507E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44283E-03 0.00568  2.80783E-04 0.01990  8.50860E-04 0.01076  6.33316E-04 0.01345  1.34914E-03 0.00909  2.76906E-04 0.01913  5.18253E-05 0.04839 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40778E-01 0.01517  1.24441E-02 0.00251  3.23259E-02 9.6E-05  1.06165E-01 0.00059  2.97933E-01 0.00029  1.23750E+00 0.00033  4.35415E+00 0.04419 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23416E-03 0.00851  2.62784E-04 0.03052  7.90507E-04 0.01695  6.03474E-04 0.01987  1.26682E-03 0.01290  2.60433E-04 0.02995  5.01400E-05 0.07231 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48176E-01 0.02538  1.24750E-02 4.3E-05  3.23256E-02 0.00014  1.06287E-01 0.00088  2.97852E-01 0.00039  1.23809E+00 0.00044  6.43546E+00 0.02915 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.91567E-07 0.00125  8.91176E-07 0.00126  1.01109E-06 0.01971 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.43140E-07 0.00116  8.42768E-07 0.00116  9.56566E-07 0.01984 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19830E-03 0.00885  2.72321E-04 0.03301  7.77218E-04 0.01787  5.80019E-04 0.02232  1.25792E-03 0.01407  2.57486E-04 0.03288  5.33295E-05 0.07566 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.50402E-01 0.02599  1.24760E-02 5.0E-05  3.23212E-02 0.00016  1.06374E-01 0.00126  2.97765E-01 0.00047  1.23785E+00 0.00057  6.35717E+00 0.04128 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.90368E-07 0.01711  7.90009E-07 0.01711  8.43954E-07 0.05880 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.47683E-07 0.01711  7.47344E-07 0.01711  7.98785E-07 0.05886 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.99812E-03 0.04212  3.04197E-04 0.12507  7.40896E-04 0.08036  5.19496E-04 0.09482  1.13279E-03 0.06093  2.63077E-04 0.12676  3.76660E-05 0.27357 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.26342E-01 0.07556  1.24793E-02 3.1E-05  3.23268E-02 0.00052  1.06270E-01 0.00326  2.97551E-01 0.00160  1.23682E+00 0.00151  6.11037E+00 0.12631 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99494E-03 0.04097  2.95894E-04 0.12520  7.39872E-04 0.07803  5.26904E-04 0.09298  1.11700E-03 0.06065  2.76661E-04 0.12620  3.86035E-05 0.26467 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.31950E-01 0.07647  1.24793E-02 3.1E-05  3.23278E-02 0.00052  1.06306E-01 0.00329  2.97580E-01 0.00160  1.23674E+00 0.00148  6.10951E+00 0.12634 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.80354E+03 0.03818 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.86858E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.38694E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22066E-03 0.00613 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.63246E+03 0.00620 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92683E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75556E-05 0.00822  6.73453E-05 0.00856  1.66680E-06 0.33851 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.04758E-05 0.02912  6.04120E-05 0.02918  9.49442E-07 0.52383 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.75230E-04 0.02429  1.74934E-04 0.02449  2.76313E-04 0.33290 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75139E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51363E+01 0.00026  4.08787E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 08:59:39 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01626E+00  9.95589E-01  9.94283E-01  9.99123E-01  9.91846E-01  9.96184E-01  9.94322E-01  1.00434E+00  1.00046E+00  1.00889E+00  9.95870E-01  1.00012E+00  1.00608E+00  9.97222E-01  9.98659E-01  1.00201E+00  9.93081E-01  1.00567E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95677E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70432E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39254E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52963E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60416E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53595E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52680E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.29840E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25264E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500444 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50047E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50047E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.03238E+03 ;
RUNNING_TIME              (idx, 1)        =  2.27441E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.12183E-01  8.72000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.25660E+02  3.63922E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  8.11667E-02  1.37500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.52909E+01  4.56657E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.22920E+02  5.08084E+02 ];
CPU_USAGE                 (idx, 1)        = 8.93586 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98502E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.83342E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.61661E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67107E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.31043E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97196E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37159E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61226E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50151E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26596E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.92103E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.67665E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.55295E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93748E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06484E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06980E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30827E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30310E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50133E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.82931E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78614E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11596E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55221E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.87606E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.84451E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.01468E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43283E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.81311E-03 -1.28352E+27  2.22081E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.80290E-01 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.63349E+18 0.00405  1.74116E-02 0.00396 ];
U233_FISS                 (idx, [1:   4]) = [  9.03888E+19 0.00050  9.63560E-01 9.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.25988E+17 0.00562  8.80546E-03 0.00561 ];
PU239_FISS                (idx, [1:   4]) = [  2.53066E+13 1.00000  2.71739E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  9.96632E+19 0.00052  7.27985E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32022E+19 0.00140  9.64355E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  2.92577E+17 0.00908  2.13718E-03 0.00907 ];
SM149_CAPT                (idx, [1:   4]) = [  1.64777E+17 0.01230  1.20372E-03 0.01231 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001883 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14023E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001883 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5391604 5.39694E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3694650 3.69804E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 915629 9.16416E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001883 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.73809E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36354E+20 3.8E-06  2.36354E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37626E+19 1.1E-07  9.37626E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36938E+20 0.00030  1.22470E+20 0.00020  1.44683E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30701E+20 0.00018  2.16232E+20 0.00011  1.44683E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53670E+20 0.00035  2.53670E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.39028E+22 0.00025  3.21603E+22 0.00022  1.74242E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.32479E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53948E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14954E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50056E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50056E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.61514E-01 0.05098 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.46128E-01 0.01122 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.91601E-04 0.02438 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.05022E+03 0.02362 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08362E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.31564E-01 0.03961 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.73689E-01 0.03961 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52077E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.32131E-01 0.00041  5.80735E-02 0.00041  1.87143E-04 0.00855 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.31809E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.31785E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.31809E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02582E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05250E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05224E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29833E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29843E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.57932E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.58252E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.53147E-03 0.00525  2.83649E-04 0.01824  8.58000E-04 0.01097  6.52311E-04 0.01321  1.38191E-03 0.00865  2.95608E-04 0.01853  5.99958E-05 0.04376 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.56837E-01 0.01590  1.24756E-02 2.8E-05  3.23101E-02 0.00011  1.06242E-01 0.00059  2.98177E-01 0.00028  1.23628E+00 0.00254  4.65728E+00 0.03917 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29531E-03 0.00764  2.68573E-04 0.02792  8.12594E-04 0.01643  6.04947E-04 0.01950  1.27665E-03 0.01206  2.75264E-04 0.02765  5.72770E-05 0.06244 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.62119E-01 0.02251  1.24756E-02 3.6E-05  3.23067E-02 0.00015  1.06207E-01 0.00087  2.98208E-01 0.00040  1.23945E+00 0.00051  6.44094E+00 0.02664 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.05279E-07 0.00115  9.04884E-07 0.00115  1.03021E-06 0.01798 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.43783E-07 0.00108  8.43415E-07 0.00108  9.60212E-07 0.01798 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21280E-03 0.00867  2.60504E-04 0.03023  7.72222E-04 0.01869  5.97486E-04 0.02174  1.25380E-03 0.01453  2.67913E-04 0.02938  6.08726E-05 0.06771 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.70983E-01 0.02769  1.24754E-02 5.4E-05  3.23080E-02 0.00019  1.06197E-01 0.00114  2.98292E-01 0.00048  1.24070E+00 0.00075  6.35781E+00 0.03792 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.02378E-07 0.01710  8.01718E-07 0.01710  8.99754E-07 0.05804 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.48086E-07 0.01709  7.47470E-07 0.01709  8.38777E-07 0.05810 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94660E-03 0.04382  1.65411E-04 0.15243  7.12864E-04 0.08393  5.58291E-04 0.08873  1.22021E-03 0.06258  2.31701E-04 0.14004  5.81242E-05 0.30767 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.65518E-01 0.10214  1.24762E-02 0.00015  3.23155E-02 0.00053  1.06325E-01 0.00340  2.98153E-01 0.00171  1.23830E+00 0.00230  6.13714E+00 0.10672 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.95748E-03 0.04301  1.66442E-04 0.15693  7.14847E-04 0.08185  5.68212E-04 0.08583  1.22145E-03 0.06233  2.24336E-04 0.13672  6.21898E-05 0.29051 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.71233E-01 0.10414  1.24762E-02 0.00015  3.23149E-02 0.00053  1.06336E-01 0.00336  2.98080E-01 0.00168  1.23830E+00 0.00230  6.13714E+00 0.10672 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.70698E+03 0.04078 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.99394E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.38299E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19823E-03 0.00559 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.55632E+03 0.00559 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94442E-09 0.00084 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76255E-05 0.00905  6.76861E-05 0.00908  1.18913E-06 0.35782 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.83471E-05 0.02658  5.84427E-05 0.02652  6.74265E-07 0.47166 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84655E-04 0.02363  1.84507E-04 0.02364  2.23726E-04 0.35112 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70736E+01 0.01158 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52680E+01 0.00026  4.09021E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 09:36:40 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01417E+00  9.99989E-01  9.95874E-01  9.99806E-01  9.97834E-01  9.99016E-01  1.00095E+00  1.00527E+00  9.97429E-01  9.95528E-01  9.97141E-01  9.94502E-01  9.95678E-01  1.00585E+00  9.96997E-01  9.99153E-01  1.00135E+00  1.00346E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95998E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70400E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38884E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52615E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60105E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54214E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53301E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.32246E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25670E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500477 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50046E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50046E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.35948E+03 ;
RUNNING_TIME              (idx, 1)        =  2.64453E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.02750E-01  9.05667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.62541E+02  3.68813E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.48667E-02  1.37000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.99543E+01  4.66332E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.59837E+02  5.15312E+02 ];
CPU_USAGE                 (idx, 1)        = 8.92211 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98503E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.81799E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.69562E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68406E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.04605E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00205E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38702E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.66050E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51235E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75024E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22169E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.11224E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.81023E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.22819E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33748E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32928E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30375E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29629E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49381E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.55106E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27944E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.15330E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.65733E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84183E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.93180E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03260E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.18531E-02 -2.61713E+27  2.23414E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01920E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.63384E+18 0.00408  1.74445E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  8.79768E+19 0.00053  9.39381E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.33081E+18 0.00330  2.48894E-02 0.00331 ];
PU239_FISS                (idx, [1:   4]) = [  9.79375E+14 0.16717  1.04515E-05 0.16738 ];
PU240_FISS                (idx, [1:   4]) = [  5.13186E+13 0.70622  5.46963E-07 0.70622 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00889E+20 0.00054  7.14591E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28961E+19 0.00142  9.13452E-02 0.00139 ];
U235_CAPT                 (idx, [1:   4]) = [  8.33473E+17 0.00530  5.90342E-03 0.00528 ];
PU239_CAPT                (idx, [1:   4]) = [  2.31845E+14 0.32998  1.64712E-06 0.33000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74985E+17 0.01212  1.23948E-03 0.01212 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001822 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14263E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001822 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5463662 5.46908E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3624506 3.62792E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 913654 9.14426E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001822 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.96161E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36256E+20 3.8E-06  2.36256E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37406E+19 1.1E-07  9.37406E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.41065E+20 0.00029  1.26282E+20 0.00022  1.47829E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34806E+20 0.00018  2.20023E+20 0.00012  1.47829E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58150E+20 0.00036  2.58150E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45589E+22 0.00025  3.27824E+22 0.00023  1.77643E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36067E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58412E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17147E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.48176E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.48176E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.16237E-01 0.04951 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.02339E-01 0.01408 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.89520E-04 0.02377 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.20551E+03 0.02203 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08560E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.41609E-01 0.03775 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.82836E-01 0.03775 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52031E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99748E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.14151E-01 0.00043  5.69551E-02 0.00043  1.91527E-04 0.00907 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.15325E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.15233E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.15325E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00746E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02309E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02442E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33709E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33505E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.68015E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.67838E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.68987E-03 0.00583  2.87737E-04 0.02012  9.04593E-04 0.01095  6.59955E-04 0.01294  1.45671E-03 0.00913  3.19679E-04 0.01871  6.11991E-05 0.04308 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.63811E-01 0.01522  1.24449E-02 0.00251  3.22950E-02 0.00012  1.06328E-01 0.00061  2.98881E-01 0.00027  1.23819E+00 0.00358  4.80177E+00 0.03862 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.37085E-03 0.00823  2.56392E-04 0.02996  8.27872E-04 0.01754  6.10594E-04 0.01841  1.32137E-03 0.01278  2.98149E-04 0.02845  5.64709E-05 0.06319 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.66707E-01 0.02292  1.24760E-02 3.7E-05  3.22871E-02 0.00015  1.06331E-01 0.00082  2.98996E-01 0.00043  1.24476E+00 0.00070  6.48907E+00 0.02596 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.18104E-07 0.00127  9.17721E-07 0.00128  1.02969E-06 0.01701 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39219E-07 0.00118  8.38869E-07 0.00119  9.41313E-07 0.01703 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.36031E-03 0.00927  2.60533E-04 0.03192  8.05910E-04 0.01797  6.03748E-04 0.02134  1.33887E-03 0.01450  3.01243E-04 0.03086  5.00008E-05 0.07648 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.56315E-01 0.02456  1.24764E-02 5.1E-05  3.22943E-02 0.00019  1.06288E-01 0.00107  2.98820E-01 0.00050  1.24208E+00 0.00088  6.69569E+00 0.03894 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.14037E-07 0.01708  8.13898E-07 0.01708  8.05267E-07 0.05040 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.44197E-07 0.01707  7.44070E-07 0.01707  7.35958E-07 0.05030 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06861E-03 0.04178  1.82351E-04 0.14288  7.20814E-04 0.07716  5.83482E-04 0.09141  1.28648E-03 0.06000  2.47394E-04 0.12892  4.80886E-05 0.29692 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52294E-01 0.08482  1.24777E-02 0.00011  3.22582E-02 0.00058  1.06290E-01 0.00302  2.97733E-01 0.00153  1.24300E+00 0.00266  7.89701E+00 0.08511 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.11284E-03 0.04114  1.82301E-04 0.13876  7.37012E-04 0.07513  5.87282E-04 0.08937  1.29867E-03 0.05878  2.56640E-04 0.12596  5.09353E-05 0.28183 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52529E-01 0.08245  1.24777E-02 0.00011  3.22577E-02 0.00058  1.06312E-01 0.00303  2.97788E-01 0.00154  1.24287E+00 0.00263  7.88110E+00 0.08536 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79557E+03 0.03871 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.12203E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33832E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37940E-03 0.00619 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.70591E+03 0.00628 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95648E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70956E-05 0.00830  6.71037E-05 0.00830  7.51616E-07 0.46034 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.72605E-05 0.02887  5.72455E-05 0.02889  7.61156E-07 0.66111 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.83359E-04 0.02302  1.83543E-04 0.02302  1.35663E-04 0.44832 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69928E+01 0.01103 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53301E+01 0.00025  4.06813E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 10:14:05 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02654E+00  1.00371E+00  1.00122E+00  9.96546E-01  9.95305E-01  1.00233E+00  9.97010E-01  1.00209E+00  1.00100E+00  9.97330E-01  1.00454E+00  9.87596E-01  1.00473E+00  9.91529E-01  9.93698E-01  1.00429E+00  9.93894E-01  9.96625E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.96898E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70310E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38851E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52618E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60341E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52570E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51661E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.29223E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25705E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499870 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50033E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50033E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.69037E+03 ;
RUNNING_TIME              (idx, 1)        =  3.01883E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.96050E-01  9.33000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.99839E+02  3.72983E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.08350E-01  1.34833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.46592E+01  4.70493E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.97226E+02  5.19155E+02 ];
CPU_USAGE                 (idx, 1)        = 8.91197 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98562E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.80666E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.87208E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73033E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59203E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01786E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39226E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82035E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55737E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.13617E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77437E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.05520E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20421E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04575E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84701E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38200E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37919E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34811E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56806E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.83311E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30563E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26145E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91676E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.45143E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09848E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03888E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.24259E-02 -4.95157E+27  2.25749E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02785E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.60536E+18 0.00405  1.71314E-02 0.00399 ];
U233_FISS                 (idx, [1:   4]) = [  8.46668E+19 0.00053  9.03560E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.71251E+18 0.00228  5.02921E-02 0.00223 ];
U238_FISS                 (idx, [1:   4]) = [  2.57364E+13 1.00000  2.73613E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  2.18808E+16 0.03239  2.33601E-04 0.03241 ];
PU240_FISS                (idx, [1:   4]) = [  4.93075E+14 0.23624  5.25879E-06 0.23631 ];
PU241_FISS                (idx, [1:   4]) = [  4.94099E+14 0.22420  5.29070E-06 0.22420 ];
TH232_CAPT                (idx, [1:   4]) = [  9.87300E+19 0.00055  6.92091E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23790E+19 0.00135  8.67795E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  1.67576E+18 0.00372  1.17478E-02 0.00373 ];
U238_CAPT                 (idx, [1:   4]) = [  1.54763E+14 0.40573  1.08094E-06 0.40574 ];
PU239_CAPT                (idx, [1:   4]) = [  1.05697E+16 0.04790  7.40787E-05 0.04790 ];
PU240_CAPT                (idx, [1:   4]) = [  1.94561E+15 0.11719  1.36447E-05 0.11711 ];
PU241_CAPT                (idx, [1:   4]) = [  1.03578E+14 0.49815  7.29200E-07 0.49814 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82343E+17 0.01273  1.27824E-03 0.01273 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001324 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10777E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001324 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487039 5.49263E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3604516 3.60791E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909769 9.10543E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001324 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36130E+20 4.3E-06  2.36130E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37022E+19 1.2E-07  9.37022E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42649E+20 0.00030  1.27800E+20 0.00021  1.48495E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36351E+20 0.00018  2.21502E+20 0.00012  1.48495E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59721E+20 0.00035  2.59721E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46891E+22 0.00026  3.29037E+22 0.00023  1.78541E+21 0.00196 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36498E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60001E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17430E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42871E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42871E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.35357E-01 0.05446 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.20261E-01 0.01311 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.85003E-04 0.02396 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.98346E+03 0.02415 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08948E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.70564E-01 0.04349 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.18529E-01 0.04349 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52000E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99830E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09281E-01 0.00045  5.66335E-02 0.00045  1.90905E-04 0.00910 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09225E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09211E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09225E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00031E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.96723E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.96613E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.41394E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.41520E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81325E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81447E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.81924E-03 0.00546  2.89907E-04 0.01821  9.01036E-04 0.01116  6.98457E-04 0.01286  1.52329E-03 0.00848  3.33531E-04 0.01810  7.30117E-05 0.03828 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.90605E-01 0.01537  1.24769E-02 3.7E-05  3.22677E-02 0.00013  1.06508E-01 0.00055  3.00121E-01 0.00030  1.25327E+00 0.00075  5.61818E+00 0.03108 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42951E-03 0.00796  2.59556E-04 0.02725  7.95273E-04 0.01713  6.19434E-04 0.01810  1.38355E-03 0.01290  3.05350E-04 0.02628  6.63495E-05 0.05869 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.98634E-01 0.02346  1.24774E-02 3.5E-05  3.22586E-02 0.00017  1.06636E-01 0.00088  3.00121E-01 0.00044  1.25278E+00 0.00091  6.87819E+00 0.02224 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.97067E-07 0.00122  8.96652E-07 0.00122  1.01834E-06 0.01571 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.15620E-07 0.00113  8.15244E-07 0.00114  9.25715E-07 0.01567 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.34808E-03 0.00929  2.54949E-04 0.03143  7.83247E-04 0.01877  6.17895E-04 0.02141  1.32943E-03 0.01451  2.95946E-04 0.03264  6.66170E-05 0.06396 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.97448E-01 0.02550  1.24763E-02 5.0E-05  3.22679E-02 0.00022  1.06551E-01 0.00100  3.00116E-01 0.00056  1.25357E+00 0.00127  6.94124E+00 0.03108 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.92085E-07 0.01708  7.91785E-07 0.01708  8.50503E-07 0.06033 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20324E-07 0.01707  7.20050E-07 0.01707  7.73769E-07 0.06042 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14605E-03 0.03958  2.41920E-04 0.12380  7.74936E-04 0.07923  5.40597E-04 0.09916  1.29412E-03 0.06044  2.47449E-04 0.14024  4.70308E-05 0.32067 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.89182E-01 0.12898  1.24768E-02 0.00013  3.22469E-02 0.00070  1.06021E-01 0.00285  2.99582E-01 0.00174  1.25563E+00 0.00356  8.42741E+00 0.06690 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13373E-03 0.03856  2.34615E-04 0.12073  7.73267E-04 0.07705  5.46895E-04 0.09816  1.28925E-03 0.05840  2.43029E-04 0.13523  4.66770E-05 0.31147 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.91667E-01 0.12912  1.24768E-02 0.00013  3.22453E-02 0.00070  1.06040E-01 0.00286  2.99549E-01 0.00173  1.25539E+00 0.00355  8.45177E+00 0.06610 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.01164E+03 0.03636 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.91180E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10267E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35447E-03 0.00577 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.76364E+03 0.00567 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92147E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70270E-05 0.00978  6.70744E-05 0.00977  1.23130E-06 0.37556 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65686E-05 0.02727  5.65438E-05 0.02729  1.27091E-06 0.44514 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77453E-04 0.02305  1.77328E-04 0.02316  2.18411E-04 0.35157 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69115E+01 0.01132 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51661E+01 0.00028  4.01878E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 10:52:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01739E+00  1.00394E+00  1.00172E+00  9.93936E-01  9.93081E-01  1.00236E+00  9.93682E-01  9.98385E-01  9.99802E-01  9.95295E-01  9.98933E-01  1.00195E+00  9.96308E-01  9.99730E-01  9.99750E-01  9.96497E-01  1.00112E+00  1.00612E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.97528E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70247E-01 9.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38687E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52481E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60466E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51425E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50512E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27612E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25865E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500537 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50043E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50043E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.02508E+03 ;
RUNNING_TIME              (idx, 1)        =  3.39822E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.88050E-01  9.20000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.37642E+02  3.78027E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.23000E-01  1.46500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.94995E+01  4.84022E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.35027E+02  5.22052E+02 ];
CPU_USAGE                 (idx, 1)        = 8.90197 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98581E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.79553E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.90307E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73854E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.70417E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01266E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38810E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85629E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56578E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.08805E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98971E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.91292E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.39188E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13500E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04261E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81855E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38587E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35181E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57386E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88340E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73577E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27630E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97481E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.89156E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11848E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03837E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.75270E-02 -6.07788E+27  2.26875E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02580E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.57578E+18 0.00426  1.68259E-02 0.00425 ];
U233_FISS                 (idx, [1:   4]) = [  8.35341E+19 0.00050  8.91950E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  5.43217E+18 0.00213  5.80033E-02 0.00208 ];
U238_FISS                 (idx, [1:   4]) = [  2.63423E+13 1.00000  2.74816E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  4.82855E+16 0.02215  5.15704E-04 0.02215 ];
PU240_FISS                (idx, [1:   4]) = [  1.45440E+15 0.12921  1.55212E-05 0.12921 ];
PU241_FISS                (idx, [1:   4]) = [  1.92139E+15 0.11510  2.05251E-05 0.11519 ];
TH232_CAPT                (idx, [1:   4]) = [  9.75086E+19 0.00055  6.84260E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22180E+19 0.00144  8.57412E-02 0.00141 ];
U235_CAPT                 (idx, [1:   4]) = [  1.92045E+18 0.00358  1.34776E-02 0.00360 ];
U238_CAPT                 (idx, [1:   4]) = [  1.81154E+14 0.37512  1.26816E-06 0.37513 ];
PU239_CAPT                (idx, [1:   4]) = [  2.26627E+16 0.03246  1.59032E-04 0.03245 ];
PU240_CAPT                (idx, [1:   4]) = [  5.31760E+15 0.07605  3.72895E-05 0.07607 ];
PU241_CAPT                (idx, [1:   4]) = [  4.14719E+14 0.26030  2.90304E-06 0.26036 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77775E+17 0.01263  1.24782E-03 0.01267 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001733 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10739E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001733 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5484257 5.48947E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3604407 3.60777E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 913069 9.13835E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001733 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.53903E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36119E+20 4.0E-06  2.36119E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36880E+19 1.1E-07  9.36880E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42553E+20 0.00028  1.27817E+20 0.00020  1.47357E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36241E+20 0.00017  2.21505E+20 0.00011  1.47357E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59592E+20 0.00036  2.59592E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46274E+22 0.00025  3.28562E+22 0.00022  1.77119E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37239E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59965E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17074E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.40064E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.40064E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.24834E-01 0.05128 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.51805E-01 0.01163 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.87034E-04 0.02438 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.07672E+03 0.02286 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08619E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.05183E-01 0.04045 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.49852E-01 0.04045 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52027E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99861E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09091E-01 0.00043  5.66339E-02 0.00043  1.93818E-04 0.00895 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09301E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09621E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09301E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00072E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94551E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94336E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44498E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44778E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.85835E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86191E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.86499E-03 0.00557  2.83999E-04 0.02017  9.15908E-04 0.01174  7.06780E-04 0.01290  1.54024E-03 0.00814  3.44225E-04 0.01860  7.38402E-05 0.04054 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.94024E-01 0.01515  1.24769E-02 3.7E-05  3.22621E-02 0.00014  1.06471E-01 0.00057  3.00210E-01 0.00031  1.25519E+00 0.00074  5.62899E+00 0.03111 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45768E-03 0.00787  2.46298E-04 0.02873  8.19075E-04 0.01646  6.39754E-04 0.01834  1.38280E-03 0.01176  3.05026E-04 0.02612  6.47363E-05 0.05702 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.93630E-01 0.02098  1.24761E-02 4.2E-05  3.22556E-02 0.00018  1.06455E-01 0.00072  3.00210E-01 0.00046  1.25515E+00 0.00102  7.03212E+00 0.02075 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.87599E-07 0.00118  8.87195E-07 0.00118  1.01309E-06 0.01534 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.06847E-07 0.00110  8.06480E-07 0.00110  9.20775E-07 0.01530 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41664E-03 0.00918  2.59645E-04 0.03494  8.04883E-04 0.01981  6.22672E-04 0.02107  1.36176E-03 0.01430  3.06587E-04 0.03135  6.10990E-05 0.07388 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.90703E-01 0.02637  1.24770E-02 4.4E-05  3.22628E-02 0.00023  1.06503E-01 0.00102  3.00410E-01 0.00053  1.25629E+00 0.00130  7.26129E+00 0.03055 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.87748E-07 0.01708  7.87547E-07 0.01708  8.23193E-07 0.05104 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.16303E-07 0.01707  7.16122E-07 0.01707  7.48574E-07 0.05103 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.03017E-03 0.04374  2.55771E-04 0.14496  7.34820E-04 0.08677  5.06126E-04 0.09207  1.19616E-03 0.06128  2.97713E-04 0.13065  3.95784E-05 0.34972 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.39758E-01 0.07901  1.24727E-02 0.00020  3.22518E-02 0.00076  1.06895E-01 0.00337  3.00258E-01 0.00178  1.26111E+00 0.00371  6.80691E+00 0.11990 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.04656E-03 0.04293  2.62155E-04 0.14372  7.33351E-04 0.08647  5.15610E-04 0.09356  1.20555E-03 0.05935  2.87595E-04 0.13207  4.22924E-05 0.33905 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.44323E-01 0.08281  1.24726E-02 0.00020  3.22536E-02 0.00076  1.06918E-01 0.00340  3.00351E-01 0.00177  1.26108E+00 0.00370  6.80691E+00 0.11990 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88738E+03 0.04080 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.82070E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.01825E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42576E-03 0.00641 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.88426E+03 0.00643 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91085E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68281E-05 0.00771  6.68166E-05 0.00772  1.73935E-06 0.33744 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.79641E-05 0.02896  5.78219E-05 0.02905  2.04747E-06 0.44397 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.80359E-04 0.02376  1.80055E-04 0.02376  2.60116E-04 0.31382 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62334E+01 0.01090 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50512E+01 0.00026  3.99780E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 11:30:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03210E+00  9.99188E-01  1.00008E+00  9.98254E-01  9.95333E-01  1.00476E+00  9.93922E-01  1.00270E+00  9.89682E-01  9.95738E-01  1.00392E+00  9.99966E-01  1.00061E+00  1.00060E+00  9.88578E-01  1.00591E+00  9.96366E-01  9.92295E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.97971E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70203E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38638E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52449E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60232E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50489E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49575E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26008E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25888E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500449 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50046E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50046E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.36253E+03 ;
RUNNING_TIME              (idx, 1)        =  3.78097E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.82650E-01  9.46000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.75774E+02  3.81319E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.36450E-01  1.34500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.44028E+01  4.90325E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.73241E+02  5.24884E+02 ];
CPU_USAGE                 (idx, 1)        = 8.89331 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98598E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.78588E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.90398E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73934E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.62888E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99830E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37807E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87159E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56761E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.17332E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17138E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.91814E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.56387E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21201E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20648E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20356E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37816E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34480E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56488E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90781E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11801E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26937E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99884E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.22966E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09953E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03909E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.26762E-02 -7.21482E+27  2.28012E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02362E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.57452E+18 0.00379  1.67886E-02 0.00371 ];
U233_FISS                 (idx, [1:   4]) = [  8.28219E+19 0.00054  8.83159E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.91972E+18 0.00211  6.31237E-02 0.00204 ];
U238_FISS                 (idx, [1:   4]) = [  5.19359E+13 0.70624  5.50835E-07 0.70623 ];
PU239_FISS                (idx, [1:   4]) = [  8.89889E+16 0.01842  9.48922E-04 0.01840 ];
PU240_FISS                (idx, [1:   4]) = [  4.16289E+15 0.07858  4.44346E-05 0.07867 ];
PU241_FISS                (idx, [1:   4]) = [  5.11438E+15 0.07073  5.45655E-05 0.07073 ];
TH232_CAPT                (idx, [1:   4]) = [  9.65568E+19 0.00054  6.77439E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20657E+19 0.00149  8.46527E-02 0.00142 ];
U235_CAPT                 (idx, [1:   4]) = [  2.09122E+18 0.00344  1.46724E-02 0.00343 ];
U238_CAPT                 (idx, [1:   4]) = [  2.87654E+14 0.29773  2.01853E-06 0.29772 ];
PU239_CAPT                (idx, [1:   4]) = [  4.24241E+16 0.02521  2.97577E-04 0.02519 ];
PU240_CAPT                (idx, [1:   4]) = [  1.07855E+16 0.05093  7.57001E-05 0.05099 ];
PU241_CAPT                (idx, [1:   4]) = [  8.82946E+14 0.16967  6.19210E-06 0.16961 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77899E+17 0.01301  1.24832E-03 0.01305 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001857 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11126E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001857 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5481645 5.48685E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3606841 3.61010E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 913371 9.14161E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001857 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.32248E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36131E+20 4.2E-06  2.36131E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36754E+19 9.8E-08  9.36754E+19 9.8E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42643E+20 0.00029  1.27900E+20 0.00021  1.47423E+19 0.00198 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36318E+20 0.00018  2.21576E+20 0.00012  1.47423E+19 0.00198 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59772E+20 0.00034  2.59772E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46042E+22 0.00024  3.28342E+22 0.00021  1.76995E+21 0.00192 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37486E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60067E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16910E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.37317E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.37317E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.72212E-01 0.04930 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.39184E-01 0.01239 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.96874E-04 0.02471 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.98127E+03 0.02326 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08586E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.41194E-01 0.03755 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.82662E-01 0.03755 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52074E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99887E+02 9.8E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09944E-01 0.00045  5.66841E-02 0.00044  1.92821E-04 0.00951 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08987E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09037E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08987E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00044E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92314E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92578E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47771E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47345E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90464E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90169E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85746E-03 0.00544  2.85409E-04 0.02102  9.04985E-04 0.01147  6.79949E-04 0.01238  1.55889E-03 0.00847  3.54443E-04 0.01737  7.37850E-05 0.03848 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.99416E-01 0.01439  1.24442E-02 0.00251  3.22524E-02 0.00013  1.06703E-01 0.00061  3.00703E-01 0.00031  1.25784E+00 0.00083  5.75249E+00 0.03073 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46584E-03 0.00771  2.58868E-04 0.03047  8.18664E-04 0.01631  6.08535E-04 0.01794  1.39840E-03 0.01220  3.19423E-04 0.02542  6.19497E-05 0.05979 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.95756E-01 0.02407  1.24756E-02 3.6E-05  3.22502E-02 0.00020  1.06761E-01 0.00086  3.00714E-01 0.00046  1.25961E+00 0.00110  6.98709E+00 0.02164 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80052E-07 0.00128  8.79604E-07 0.00128  1.01343E-06 0.01758 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.00746E-07 0.00123  8.00338E-07 0.00123  9.22169E-07 0.01759 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.38483E-03 0.00952  2.50505E-04 0.03540  8.02271E-04 0.01906  5.84770E-04 0.02378  1.36716E-03 0.01478  3.12465E-04 0.02891  6.76572E-05 0.06512 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.10702E-01 0.02492  1.24763E-02 4.6E-05  3.22462E-02 0.00025  1.06854E-01 0.00114  3.00741E-01 0.00054  1.25998E+00 0.00166  7.33613E+00 0.02828 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.77101E-07 0.01720  7.76635E-07 0.01720  8.80113E-07 0.04806 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.07154E-07 0.01719  7.06730E-07 0.01719  8.00681E-07 0.04800 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.19003E-03 0.04112  2.68489E-04 0.15109  8.20424E-04 0.07786  5.55139E-04 0.08468  1.19226E-03 0.06562  2.84688E-04 0.13228  6.90345E-05 0.25871 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.04580E-01 0.11182  1.24777E-02 0.00012  3.22916E-02 0.00068  1.07280E-01 0.00352  3.00117E-01 0.00183  1.26412E+00 0.00445  6.97630E+00 0.08943 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.18163E-03 0.04031  2.66132E-04 0.14513  8.09025E-04 0.07634  5.61148E-04 0.08525  1.18849E-03 0.06369  2.81547E-04 0.13403  7.52897E-05 0.25399 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.08114E-01 0.10943  1.24777E-02 0.00012  3.22884E-02 0.00068  1.07283E-01 0.00353  3.00215E-01 0.00183  1.26401E+00 0.00443  6.97630E+00 0.08943 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.13032E+03 0.03757 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.74935E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.96079E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35468E-03 0.00576 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.83456E+03 0.00573 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91523E-09 0.00086 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67523E-05 0.00895  6.67743E-05 0.00895  6.14463E-07 0.51202 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.51767E-05 0.02791  5.52653E-05 0.02793  3.56693E-07 0.53515 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.90295E-04 0.02404  1.90634E-04 0.02404  9.97456E-05 0.49939 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62178E+01 0.01224 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49575E+01 0.00026  3.97725E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 12:09:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01070E+00  1.00148E+00  9.91432E-01  9.97945E-01  9.93183E-01  9.99585E-01  1.00150E+00  9.95280E-01  9.99774E-01  1.00543E+00  1.00164E+00  1.00418E+00  9.99951E-01  9.90537E-01  9.93496E-01  1.00608E+00  1.00088E+00  1.00693E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.99477E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70052E-01 9.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38428E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52293E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60315E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49068E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48151E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23953E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.26598E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500545 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50045E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50045E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.70542E+03 ;
RUNNING_TIME              (idx, 1)        =  4.17080E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.07923E+00  9.65667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.14612E+02  3.88384E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.49067E-01  1.26167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.94877E+01  5.08482E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.12043E+02  5.26766E+02 ];
CPU_USAGE                 (idx, 1)        = 8.88420 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98495E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.77574E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.92734E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74568E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.98220E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96485E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35830E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92837E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57590E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.73737E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75285E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.24620E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03908E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44272E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70421E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34349E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37316E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33813E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55743E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95108E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27190E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26691E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05575E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41922E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07952E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04265E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.31453E-02 -1.17343E+28  2.32531E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02120E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.54379E+18 0.00411  1.64773E-02 0.00408 ];
U233_FISS                 (idx, [1:   4]) = [  8.07327E+19 0.00055  8.61695E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.92123E+18 0.00182  7.38744E-02 0.00178 ];
U238_FISS                 (idx, [1:   4]) = [  1.03697E+14 0.49822  1.10496E-06 0.49823 ];
PU239_FISS                (idx, [1:   4]) = [  3.06894E+17 0.00904  3.27531E-03 0.00900 ];
PU240_FISS                (idx, [1:   4]) = [  2.35696E+16 0.03284  2.51553E-04 0.03287 ];
PU241_FISS                (idx, [1:   4]) = [  3.46253E+16 0.02775  3.69368E-04 0.02770 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44228E+19 0.00057  6.58718E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17655E+19 0.00145  8.20831E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.44878E+18 0.00323  1.70834E-02 0.00319 ];
U238_CAPT                 (idx, [1:   4]) = [  1.25005E+15 0.14198  8.73495E-06 0.14196 ];
PU239_CAPT                (idx, [1:   4]) = [  1.45769E+17 0.01252  1.01685E-03 0.01250 ];
PU240_CAPT                (idx, [1:   4]) = [  5.73664E+16 0.02240  4.00153E-04 0.02239 ];
PU241_CAPT                (idx, [1:   4]) = [  7.42199E+15 0.05639  5.17647E-05 0.05633 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82278E+17 0.01184  1.27155E-03 0.01182 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001814 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10176E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001814 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5493855 5.49919E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3591280 3.59437E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 916679 9.17452E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001814 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 0.00000E+00 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36285E+20 4.6E-06  2.36285E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36358E+19 1.0E-07  9.36358E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43315E+20 0.00029  1.28349E+20 0.00020  1.49662E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36951E+20 0.00017  2.21984E+20 0.00011  1.49662E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60663E+20 0.00035  2.60663E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46962E+22 0.00026  3.29016E+22 0.00023  1.79451E+21 0.00183 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39156E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60866E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16938E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26748E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26748E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.39709E-01 0.05216 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.22829E-01 0.01260 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.02906E-04 0.02481 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.85143E+03 0.02456 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08257E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.01979E-01 0.04067 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.46745E-01 0.04067 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52345E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99972E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.06877E-01 0.00045  5.64898E-02 0.00044  1.96645E-04 0.00922 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06780E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06525E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06780E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98408E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.87970E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.87989E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.54327E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.54266E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.99118E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.99944E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.95118E-03 0.00575  2.81727E-04 0.02050  9.36774E-04 0.01064  7.08460E-04 0.01212  1.58037E-03 0.00914  3.61531E-04 0.01635  8.23198E-05 0.04142 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08788E-01 0.01594  1.24449E-02 0.00251  3.22322E-02 0.00015  1.06661E-01 0.00056  3.01094E-01 0.00030  1.26005E+00 0.00093  5.64791E+00 0.03110 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.50449E-03 0.00785  2.46191E-04 0.02976  8.22506E-04 0.01570  6.28632E-04 0.01809  1.40599E-03 0.01260  3.27996E-04 0.02295  7.31754E-05 0.05486 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.16185E-01 0.02236  1.24764E-02 7.1E-05  3.22297E-02 0.00019  1.06653E-01 0.00080  3.01063E-01 0.00044  1.25872E+00 0.00115  6.99675E+00 0.02074 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64939E-07 0.00129  8.64386E-07 0.00129  1.03776E-06 0.03361 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.84322E-07 0.00119  7.83821E-07 0.00119  9.40610E-07 0.03336 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.47440E-03 0.00926  2.34765E-04 0.03438  8.16332E-04 0.01873  6.11786E-04 0.02254  1.41684E-03 0.01398  3.22181E-04 0.02682  7.24904E-05 0.06416 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14764E-01 0.02555  1.24748E-02 6.8E-05  3.22333E-02 0.00027  1.06792E-01 0.00111  3.00812E-01 0.00049  1.26098E+00 0.00175  6.93785E+00 0.02972 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63139E-07 0.01707  7.62630E-07 0.01707  8.67117E-07 0.05772 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.92073E-07 0.01706  6.91613E-07 0.01706  7.85903E-07 0.05743 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01594E-03 0.04485  2.04147E-04 0.17744  6.85919E-04 0.08201  5.17705E-04 0.10034  1.25510E-03 0.06121  3.11922E-04 0.12820  4.11449E-05 0.23755 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.09284E-01 0.09931  1.24752E-02 0.00018  3.22137E-02 0.00073  1.06488E-01 0.00292  3.01150E-01 0.00189  1.26325E+00 0.00676  7.08713E+00 0.09112 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99725E-03 0.04425  2.09941E-04 0.17643  6.65334E-04 0.08003  5.09646E-04 0.09917  1.26445E-03 0.06088  3.06720E-04 0.12787  4.11636E-05 0.23850 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.04939E-01 0.09612  1.24751E-02 0.00018  3.22171E-02 0.00072  1.06490E-01 0.00292  3.01165E-01 0.00188  1.26308E+00 0.00676  7.08547E+00 0.09110 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.97968E+03 0.04192 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.58505E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78493E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46526E-03 0.00587 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03635E+03 0.00581 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90866E-09 0.00085 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69007E-05 0.00847  6.69111E-05 0.00847  8.32730E-07 0.41344 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.66825E-05 0.02725  5.66380E-05 0.02726  1.07418E-06 0.58928 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.94194E-04 0.02388  1.94364E-04 0.02383  1.58818E-04 0.40793 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60945E+01 0.01186 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48151E+01 0.00026  3.93446E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 12:48:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01695E+00  9.99345E-01  9.96203E-01  1.00700E+00  9.99090E-01  1.00115E+00  9.87777E-01  9.95785E-01  9.95445E-01  1.00720E+00  9.99116E-01  9.96876E-01  9.97842E-01  1.00739E+00  9.90690E-01  1.01112E+00  9.94171E-01  9.96843E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.03919E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69608E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36991E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51045E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62004E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48808E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47886E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27982E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.29756E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500440 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50031E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50031E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.04845E+03 ;
RUNNING_TIME              (idx, 1)        =  4.56101E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.18177E+00  1.02533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.53479E+02  3.88665E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.61867E-01  1.28000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.45919E+01  5.10413E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.51049E+02  5.28956E+02 ];
CPU_USAGE                 (idx, 1)        = 8.87621 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98556E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.76692E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.91451E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74035E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.61977E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92257E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33688E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95784E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57275E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53980E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39218E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36506E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14208E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69294E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23937E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58193E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34689E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31704E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52917E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.98635E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59923E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23470E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03523E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.15604E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01672E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04535E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06427E-01 -2.34988E+28  2.44296E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01877E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.51982E+18 0.00426  1.62438E-02 0.00421 ];
U233_FISS                 (idx, [1:   4]) = [  7.93099E+19 0.00057  8.47683E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.15293E+18 0.00188  7.64537E-02 0.00183 ];
U238_FISS                 (idx, [1:   4]) = [  1.81697E+14 0.37513  1.95510E-06 0.37513 ];
PU239_FISS                (idx, [1:   4]) = [  6.14511E+17 0.00653  6.56833E-03 0.00652 ];
PU240_FISS                (idx, [1:   4]) = [  7.23579E+16 0.01802  7.73086E-04 0.01796 ];
PU241_FISS                (idx, [1:   4]) = [  1.26353E+17 0.01383  1.35028E-03 0.01380 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26724E+19 0.00054  6.43883E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15516E+19 0.00148  8.02610E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51936E+18 0.00317  1.75046E-02 0.00315 ];
U238_CAPT                 (idx, [1:   4]) = [  3.08340E+15 0.09277  2.13963E-05 0.09282 ];
PU239_CAPT                (idx, [1:   4]) = [  2.93223E+17 0.00993  2.03738E-03 0.00992 ];
PU240_CAPT                (idx, [1:   4]) = [  1.91294E+17 0.01119  1.32936E-03 0.01122 ];
PU241_CAPT                (idx, [1:   4]) = [  2.74376E+16 0.03002  1.90700E-04 0.03005 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82713E+17 0.01252  1.26980E-03 0.01257 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001245 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10040E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001245 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5501753 5.50736E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3576757 3.58009E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 922735 9.23546E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001245 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.55182E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36592E+20 4.9E-06  2.36592E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35952E+19 1.1E-07  9.35952E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43962E+20 0.00027  1.28651E+20 0.00019  1.53106E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37557E+20 0.00017  2.22247E+20 0.00011  1.53106E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61338E+20 0.00033  2.61338E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49991E+22 0.00023  3.31781E+22 0.00022  1.82103E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41367E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61694E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17172E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.05841E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.05841E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.95755E-01 0.04703 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19236E-01 0.01086 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.65222E-04 0.02047 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.61289E+03 0.02337 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07649E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.87788E-01 0.03356 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.24289E-01 0.03356 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52782E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00059E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04908E-01 0.00045  5.63675E-02 0.00044  1.95141E-04 0.00860 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05068E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05349E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05068E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97133E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85189E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85051E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58687E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58863E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07373E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.06947E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93598E-03 0.00558  2.86898E-04 0.01971  9.31247E-04 0.01119  6.95286E-04 0.01270  1.58251E-03 0.00872  3.57318E-04 0.01787  8.27219E-05 0.03658 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08769E-01 0.01458  1.24828E-02 0.00013  3.22044E-02 0.00015  1.06760E-01 0.00054  3.01309E-01 0.00030  1.25874E+00 0.00095  5.92857E+00 0.02833 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51697E-03 0.00792  2.55187E-04 0.02935  8.17308E-04 0.01527  6.24519E-04 0.01818  1.42517E-03 0.01210  3.24078E-04 0.02617  7.07140E-05 0.05364 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.06045E-01 0.02031  1.24800E-02 9.2E-05  3.22032E-02 0.00020  1.06794E-01 0.00077  3.01287E-01 0.00042  1.25969E+00 0.00134  7.00152E+00 0.02002 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.60669E-07 0.00135  8.60081E-07 0.00135  1.02815E-06 0.01923 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.78778E-07 0.00131  7.78246E-07 0.00130  9.30269E-07 0.01922 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44447E-03 0.00856  2.52518E-04 0.03398  8.22595E-04 0.01839  5.97772E-04 0.02190  1.38622E-03 0.01377  3.10033E-04 0.02876  7.53305E-05 0.06252 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14948E-01 0.02571  1.24834E-02 0.00023  3.21943E-02 0.00025  1.06761E-01 0.00104  3.01267E-01 0.00054  1.26085E+00 0.00172  7.09174E+00 0.02803 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.56840E-07 0.01713  7.56427E-07 0.01713  8.35935E-07 0.05414 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.84992E-07 0.01712  6.84619E-07 0.01712  7.56633E-07 0.05412 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00251E-03 0.04098  2.32458E-04 0.13084  7.29821E-04 0.07647  5.52032E-04 0.09143  1.20659E-03 0.06513  2.17884E-04 0.14183  6.37267E-05 0.28224 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.93498E-01 0.10056  1.24900E-02 0.00103  3.22016E-02 0.00088  1.06637E-01 0.00288  3.00994E-01 0.00190  1.25608E+00 0.00387  7.22896E+00 0.09868 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00310E-03 0.04048  2.36500E-04 0.12985  7.37352E-04 0.07438  5.50234E-04 0.09070  1.19999E-03 0.06443  2.19247E-04 0.13559  5.97825E-05 0.28244 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.88906E-01 0.09908  1.24900E-02 0.00103  3.22013E-02 0.00088  1.06664E-01 0.00289  3.00916E-01 0.00189  1.25605E+00 0.00385  7.24464E+00 0.09828 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.99089E+03 0.03762 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.53616E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.72399E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39743E-03 0.00563 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.98171E+03 0.00577 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.02889E-09 0.00090 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79374E-05 0.00596  6.79131E-05 0.00597  1.05449E-06 0.41178 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.85536E-05 0.02149  5.84967E-05 0.02155  1.11562E-06 0.51732 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.53530E-04 0.02016  2.53931E-04 0.02020  1.59534E-04 0.40734 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63842E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47886E+01 0.00027  3.90814E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 13:27:30 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01804E+00  9.98134E-01  9.99538E-01  9.95906E-01  9.94227E-01  1.00463E+00  9.98081E-01  9.94887E-01  9.97088E-01  9.99597E-01  9.92738E-01  9.94874E-01  1.00497E+00  1.00518E+00  9.98029E-01  1.00577E+00  1.00041E+00  9.97898E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.08127E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69187E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35605E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49829E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63402E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49471E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48546E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33645E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.33189E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500536 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50027E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50027E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.39318E+03 ;
RUNNING_TIME              (idx, 1)        =  4.95292E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.27998E+00  9.82167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.92513E+02  3.90346E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.76850E-01  1.49833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.96897E+01  5.09778E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.90241E+02  5.29221E+02 ];
CPU_USAGE                 (idx, 1)        = 8.86988 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98553E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.75996E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.91201E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73853E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.41523E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91233E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33192E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96543E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57126E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64085E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56669E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45859E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16014E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76489E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39519E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93195E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33906E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31126E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52110E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00339E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00902E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22500E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02371E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34905E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00020E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04668E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61026E-01 -3.55541E+28  2.56351E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01505E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.51053E+18 0.00423  1.61447E-02 0.00417 ];
U233_FISS                 (idx, [1:   4]) = [  7.91871E+19 0.00055  8.46411E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.11380E+18 0.00201  7.60374E-02 0.00194 ];
U238_FISS                 (idx, [1:   4]) = [  2.59678E+14 0.31267  2.78242E-06 0.31269 ];
PU239_FISS                (idx, [1:   4]) = [  6.58030E+17 0.00635  7.03321E-03 0.00632 ];
PU240_FISS                (idx, [1:   4]) = [  8.65886E+16 0.01749  9.25309E-04 0.01746 ];
PU241_FISS                (idx, [1:   4]) = [  1.57724E+17 0.01272  1.68609E-03 0.01273 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21417E+19 0.00052  6.39109E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15490E+19 0.00149  8.01053E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51827E+18 0.00317  1.74675E-02 0.00316 ];
U238_CAPT                 (idx, [1:   4]) = [  2.32876E+15 0.10985  1.61609E-05 0.10992 ];
PU239_CAPT                (idx, [1:   4]) = [  3.04429E+17 0.00960  2.11143E-03 0.00957 ];
PU240_CAPT                (idx, [1:   4]) = [  2.26255E+17 0.01113  1.56935E-03 0.01112 ];
PU241_CAPT                (idx, [1:   4]) = [  3.13865E+16 0.02798  2.17774E-04 0.02799 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85074E+17 0.01188  1.28397E-03 0.01193 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001096 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09627E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001096 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5504118 5.50972E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3571934 3.57538E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 925044 9.25861E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001096 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.77420E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36657E+20 4.9E-06  2.36657E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35889E+19 1.1E-07  9.35889E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44083E+20 0.00029  1.28316E+20 0.00019  1.57675E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37672E+20 0.00017  2.21905E+20 0.00011  1.57675E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61671E+20 0.00033  2.61671E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53056E+22 0.00024  3.34527E+22 0.00022  1.85292E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.42278E+19 0.00135 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61900E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17495E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87717E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87717E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.80032E-01 0.04238 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23091E-01 0.00937 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.07227E-04 0.01992 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.55406E+03 0.02348 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07419E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.42137E-01 0.02929 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.73336E-01 0.02929 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52869E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03948E-01 0.00045  5.63117E-02 0.00044  1.95746E-04 0.00886 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.04611E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.04447E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.04611E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96934E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84807E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84983E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59291E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58970E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08092E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07938E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94461E-03 0.00560  2.82171E-04 0.01988  9.17524E-04 0.01139  7.06789E-04 0.01394  1.59007E-03 0.00884  3.63984E-04 0.01756  8.40711E-05 0.03492 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.12613E-01 0.01482  1.24799E-02 0.00012  3.21999E-02 0.00016  1.06844E-01 0.00058  3.01273E-01 0.00032  1.25590E+00 0.00115  5.73474E+00 0.02869 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48995E-03 0.00761  2.39769E-04 0.02868  8.16893E-04 0.01643  6.31642E-04 0.01855  1.40845E-03 0.01177  3.18204E-04 0.02506  7.49962E-05 0.05475 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.13950E-01 0.02273  1.24802E-02 0.00011  3.22032E-02 0.00020  1.06858E-01 0.00077  3.01289E-01 0.00044  1.25555E+00 0.00143  6.75508E+00 0.02137 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68661E-07 0.00135  8.68274E-07 0.00136  9.76814E-07 0.01630 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.85160E-07 0.00127  7.84811E-07 0.00128  8.82936E-07 0.01627 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45348E-03 0.00892  2.44506E-04 0.03447  8.00369E-04 0.01812  6.07570E-04 0.02294  1.40290E-03 0.01458  3.21638E-04 0.02772  7.64894E-05 0.05932 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.21232E-01 0.02451  1.24829E-02 0.00024  3.22047E-02 0.00027  1.06875E-01 0.00117  3.01333E-01 0.00055  1.25605E+00 0.00213  6.83598E+00 0.02952 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.64581E-07 0.01723  7.64093E-07 0.01724  8.38276E-07 0.06291 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.91368E-07 0.01723  6.90927E-07 0.01724  7.58010E-07 0.06296 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09206E-03 0.04028  2.26610E-04 0.14097  6.58785E-04 0.08027  6.01362E-04 0.08812  1.28655E-03 0.06429  2.59163E-04 0.13323  5.95934E-05 0.22380 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.15411E-01 0.09723  1.24743E-02 0.00019  3.21915E-02 0.00096  1.07125E-01 0.00335  3.01453E-01 0.00187  1.25066E+00 0.00597  6.29504E+00 0.08988 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10303E-03 0.03951  2.30626E-04 0.13779  6.64274E-04 0.07990  5.97096E-04 0.08836  1.28961E-03 0.06266  2.61176E-04 0.13253  6.02543E-05 0.21456 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.14402E-01 0.09454  1.24742E-02 0.00019  3.21908E-02 0.00094  1.07086E-01 0.00332  3.01361E-01 0.00186  1.25094E+00 0.00584  6.29504E+00 0.08988 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.08168E+03 0.03713 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60375E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.77670E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46875E-03 0.00515 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03201E+03 0.00516 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16070E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84546E-05 0.00585  6.84726E-05 0.00587  1.08528E-06 0.38272 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.85979E-05 0.01916  5.85734E-05 0.01920  1.25513E-06 0.43621 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.96363E-04 0.01868  2.96834E-04 0.01864  1.79256E-04 0.37938 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58989E+01 0.01192 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48546E+01 0.00028  3.90407E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:12:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 13:49:05 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615132880 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03492E+00  1.00006E+00  9.99098E-01  9.96361E-01  9.98908E-01  9.95884E-01  9.97399E-01  9.93964E-01  1.00057E+00  1.00123E+00  9.99405E-01  1.00474E+00  1.00103E+00  9.95930E-01  9.91965E-01  9.97426E-01  9.95845E-01  9.95257E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.12917E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68708E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34147E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.48576E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64469E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49574E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48636E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.38424E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.36736E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499778 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50036E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50036E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.58227E+03 ;
RUNNING_TIME              (idx, 1)        =  5.16885E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.00313E+00  1.00313E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.38305E+00  1.03067E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.13944E+02  2.14307E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.90650E-01  1.38000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  9.48333E-03  9.48333E-03 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.25340E+01  2.84423E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.14092E+02  5.14092E+02 ];
CPU_USAGE                 (idx, 1)        = 8.86518 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97683E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.75517E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22581.47;
MEMSIZE                   (idx, 1)        = 22384.05;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 232.34;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 197.42;

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

TOT_ACTIVITY              (idx, 1)        =  6.90795E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73768E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.63926E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90598E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32820E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96769E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57076E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65494E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61371E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47044E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16145E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78565E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44059E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03105E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33665E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30950E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51864E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00607E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13435E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21926E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01995E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.39023E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98758E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04894E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16297E-01 -4.77578E+28  2.68555E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01058E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.51363E+18 0.00414  1.61607E-02 0.00408 ];
U233_FISS                 (idx, [1:   4]) = [  7.92651E+19 0.00057  8.46337E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.10084E+18 0.00194  7.58182E-02 0.00188 ];
U238_FISS                 (idx, [1:   4]) = [  1.58118E+14 0.40573  1.68318E-06 0.40574 ];
PU239_FISS                (idx, [1:   4]) = [  6.65687E+17 0.00602  7.10751E-03 0.00599 ];
PU240_FISS                (idx, [1:   4]) = [  8.85604E+16 0.01738  9.45902E-04 0.01740 ];
PU241_FISS                (idx, [1:   4]) = [  1.57727E+17 0.01288  1.68416E-03 0.01288 ];
TH232_CAPT                (idx, [1:   4]) = [  9.17934E+19 0.00054  6.36396E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15615E+19 0.00160  8.01534E-02 0.00152 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50082E+18 0.00301  1.73393E-02 0.00304 ];
U238_CAPT                 (idx, [1:   4]) = [  2.95510E+15 0.09252  2.05245E-05 0.09264 ];
PU239_CAPT                (idx, [1:   4]) = [  3.15054E+17 0.00921  2.18423E-03 0.00921 ];
PU240_CAPT                (idx, [1:   4]) = [  2.28756E+17 0.01035  1.58608E-03 0.01036 ];
PU241_CAPT                (idx, [1:   4]) = [  3.34410E+16 0.02914  2.31777E-04 0.02913 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84875E+17 0.01222  1.28169E-03 0.01219 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001456 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07405E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001456 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5495024 5.50043E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3568357 3.57148E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 938075 9.38827E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001456 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.65661E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36669E+20 4.9E-06  2.36669E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35882E+19 1.2E-07  9.35882E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44172E+20 0.00028  1.28044E+20 0.00018  1.61275E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37760E+20 0.00017  2.21633E+20 0.00010  1.61275E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62235E+20 0.00033  2.62235E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.56118E+22 0.00024  3.37391E+22 0.00021  1.87266E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46199E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62380E+20 0.00018 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17769E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.71830E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.71830E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.97052E-01 0.03456 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07894E-01 0.00883 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.84617E-04 0.01676 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.80462E+03 0.02637 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06123E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.34794E-01 0.02206 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.56383E-01 0.02206 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52883E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00074E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.02893E-01 0.00042  5.62546E-02 0.00041  1.94417E-04 0.00883 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.02992E-01 0.00018 ];
COL_KEFF                  (idx, [1:   2]) = [  9.02545E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.02992E-01 0.00018 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96607E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84841E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85079E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59232E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58818E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08487E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.08188E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96714E-03 0.00532  2.79515E-04 0.01982  9.28602E-04 0.01087  7.17665E-04 0.01211  1.59551E-03 0.00811  3.58679E-04 0.01746  8.71683E-05 0.03787 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.17123E-01 0.01474  1.24165E-02 0.00355  3.21970E-02 0.00015  1.06790E-01 0.00056  3.01474E-01 0.00032  1.25740E+00 0.00114  6.13233E+00 0.02717 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49618E-03 0.00773  2.47837E-04 0.02960  8.27587E-04 0.01598  6.26537E-04 0.01809  1.39515E-03 0.01191  3.21014E-04 0.02606  7.80573E-05 0.05201 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.14507E-01 0.02079  1.24784E-02 0.00011  3.21938E-02 0.00021  1.06739E-01 0.00073  3.01544E-01 0.00042  1.25872E+00 0.00148  7.11426E+00 0.01944 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80350E-07 0.00144  8.79950E-07 0.00144  9.91351E-07 0.01635 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.94807E-07 0.00138  7.94446E-07 0.00138  8.95035E-07 0.01635 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44539E-03 0.00884  2.34700E-04 0.03461  7.95952E-04 0.01892  6.20612E-04 0.02084  1.40521E-03 0.01302  3.14596E-04 0.02869  7.43199E-05 0.06374 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.20509E-01 0.02639  1.24800E-02 0.00017  3.21898E-02 0.00028  1.06695E-01 0.00100  3.01394E-01 0.00055  1.25882E+00 0.00181  7.24790E+00 0.02739 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.67981E-07 0.01729  7.67760E-07 0.01729  8.70144E-07 0.05533 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.93398E-07 0.01728  6.93198E-07 0.01728  7.85726E-07 0.05533 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11581E-03 0.04109  2.49967E-04 0.13820  6.49104E-04 0.08181  5.62312E-04 0.09230  1.32251E-03 0.06063  2.74746E-04 0.12882  5.71713E-05 0.25743 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.87472E-01 0.10515  1.24765E-02 0.00015  3.21936E-02 0.00082  1.06503E-01 0.00264  3.01661E-01 0.00196  1.26202E+00 0.00737  6.22697E+00 0.09299 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10359E-03 0.04085  2.45887E-04 0.13532  6.54880E-04 0.08005  5.49190E-04 0.09323  1.31758E-03 0.05992  2.76731E-04 0.12758  5.93243E-05 0.25135 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.88047E-01 0.10530  1.24765E-02 0.00015  3.21932E-02 0.00081  1.06510E-01 0.00263  3.01692E-01 0.00195  1.26266E+00 0.00725  6.25624E+00 0.09335 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.08342E+03 0.03710 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.71319E-07 0.00086 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.86651E-07 0.00074 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44611E-03 0.00566 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95536E+03 0.00562 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.32338E-09 0.00102 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.96887E-05 0.00502  6.96969E-05 0.00502  1.82505E-06 0.31139 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.24044E-05 0.01654  6.24365E-05 0.01655  1.53857E-06 0.38755 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.67643E-04 0.01625  3.68002E-04 0.01627  2.98349E-04 0.29887 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59476E+01 0.01151 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48636E+01 0.00027  3.90370E+01 0.00035 ];

