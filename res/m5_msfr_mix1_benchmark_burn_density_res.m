
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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 17:47:15 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03103E+00  9.91798E-01  9.89095E-01  9.93914E-01  9.90946E-01  9.94208E-01  9.99395E-01  9.96559E-01  9.91294E-01  1.00640E+00  1.00523E+00  9.92915E-01  1.00520E+00  1.00849E+00  9.93233E-01  1.00705E+00  1.00611E+00  9.97132E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.91757E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70824E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31445E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45879E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71445E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47851E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46863E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.45105E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10286E+00 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333938 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33388E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33388E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.52546E+02 ;
RUNNING_TIME              (idx, 1)        =  2.77163E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.82917E-01  1.82917E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.65461E+01  2.65461E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.67807E+00  1.63277E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.61251E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.11182 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98512E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.05098E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  2.69786E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.12165E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.05819E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.69786E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.12165E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.59302E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.37646E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.59302E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.37646E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.24125E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.05724E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09042E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.22113E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15174E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.73999E-01 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.55907E+18 0.00396  1.66204E-02 0.00388 ];
U233_FISS                 (idx, [1:   4]) = [  9.22404E+19 0.00054  9.83380E-01 6.6E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22981E+19 0.00057  7.48194E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33622E+19 0.00129  1.08321E-01 0.00125 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001646 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13056E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001646 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5119815 5.12502E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3893240 3.89690E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 988591 9.89381E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001646 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.26432E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36462E+20 3.5E-06  2.36462E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37768E+19 1.2E-07  9.37768E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23375E+20 0.00032  1.08760E+20 0.00024  1.46148E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17152E+20 0.00018  2.02537E+20 0.00013  1.46148E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40704E+20 0.00038  2.40704E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.31306E+22 0.00027  3.13458E+22 0.00024  1.78475E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.38154E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.40967E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07679E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29424E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.32814E-01 0.07660 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.92095E-01 0.00732 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.58180E-04 0.01414 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.03356E+04 0.02986 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01071E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.04652E-01 0.06224 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.54789E-01 0.06224 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52154E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.82457E-01 0.00044  1.08831E-01 0.00044  3.49432E-04 0.00950 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.82440E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.82418E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.82440E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09032E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02330E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02319E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33669E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33667E-02 0.00102 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50359E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50084E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28139E-03 0.00608  2.78239E-04 0.02028  7.92489E-04 0.01211  5.97324E-04 0.01277  1.29242E-03 0.00935  2.71333E-04 0.01986  4.95879E-05 0.04762 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.41561E-01 0.01619  1.24760E-02 2.6E-05  3.23385E-02 9.4E-05  1.05951E-01 0.00060  2.97455E-01 0.00027  1.23523E+00 0.00022  4.85022E+00 0.04157 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20777E-03 0.00905  2.74763E-04 0.02887  7.70745E-04 0.01757  5.86175E-04 0.01854  1.26739E-03 0.01343  2.56803E-04 0.02871  5.18974E-05 0.07685 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.49226E-01 0.02757  1.24764E-02 3.3E-05  3.23400E-02 0.00015  1.05829E-01 0.00068  2.97389E-01 0.00038  1.23528E+00 0.00030  6.27374E+00 0.03053 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.81693E-07 0.00129  8.81426E-07 0.00129  9.66888E-07 0.01595 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.66171E-07 0.00120  8.65909E-07 0.00120  9.49822E-07 0.01592 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19338E-03 0.00990  2.76046E-04 0.03163  7.63084E-04 0.01857  5.82999E-04 0.02100  1.24982E-03 0.01552  2.70385E-04 0.02996  5.10553E-05 0.06720 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.49502E-01 0.02306  1.24762E-02 4.3E-05  3.23429E-02 0.00015  1.05995E-01 0.00104  2.97290E-01 0.00044  1.23621E+00 0.00036  6.24546E+00 0.04164 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.81946E-07 0.01970  7.81593E-07 0.01971  8.92663E-07 0.05549 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.68484E-07 0.01970  7.68137E-07 0.01970  8.77162E-07 0.05546 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.72946E-03 0.04472  2.19680E-04 0.13581  5.87761E-04 0.08529  5.37024E-04 0.09306  1.11355E-03 0.06543  2.46798E-04 0.12496  2.46421E-05 0.31539 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.23118E-01 0.06920  1.24764E-02 0.00012  3.23265E-02 0.00046  1.05479E-01 0.00249  2.97284E-01 0.00156  1.23405E+00 0.00113  6.84269E+00 0.11690 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.74208E-03 0.04370  2.17366E-04 0.13300  5.92071E-04 0.08405  5.35285E-04 0.09031  1.12551E-03 0.06295  2.47673E-04 0.12121  2.41735E-05 0.27159 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.22439E-01 0.06422  1.24763E-02 0.00012  3.23269E-02 0.00046  1.05497E-01 0.00251  2.97332E-01 0.00156  1.23400E+00 0.00114  6.88068E+00 0.11698 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.51835E+03 0.04067 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.74778E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.59378E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.12819E-03 0.00592 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.57631E+03 0.00592 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.25797E-09 0.00128 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.24788E-05 0.00400  7.24970E-05 0.00401  3.62041E-06 0.24784 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.86050E-05 0.01449  7.87289E-05 0.01453  2.45826E-06 0.29864 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.31998E-04 0.01380  5.32039E-04 0.01384  4.82838E-04 0.25100 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73989E+01 0.01207 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46863E+01 0.00026  4.02908E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 18:30:03 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02259E+00  9.95241E-01  9.98488E-01  1.00565E+00  9.99888E-01  9.96475E-01  9.98086E-01  1.00248E+00  9.93106E-01  1.00051E+00  1.00014E+00  9.99884E-01  9.96975E-01  1.00445E+00  9.98077E-01  9.96162E-01  9.97298E-01  9.94507E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.91053E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70895E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31503E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45900E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71254E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49067E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48084E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.47336E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10277E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333914 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33374E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33374E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.55186E+02 ;
RUNNING_TIME              (idx, 1)        =  7.05216E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.01150E-01  1.18233E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.92090E+01  4.26629E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.82500E-02  1.82500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.38277E+00  2.70465E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.78595E+01  4.14733E+02 ];
CPU_USAGE                 (idx, 1)        = 9.29057 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98938E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.23648E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  5.77078E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60359E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65603E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.47684E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.04083E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26294E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46841E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.44837E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65458E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.38615E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.77476E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06222E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17710E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44216E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35393E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17401E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55174E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60495E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38118E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39289E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51569E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.52631E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.29509E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.29440E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26594E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.11315E-04 -2.39522E+25  2.15198E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.85730E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.57317E+18 0.00389  1.67804E-02 0.00389 ];
U233_FISS                 (idx, [1:   4]) = [  9.21503E+19 0.00049  9.82907E-01 6.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.44802E+14 0.31144  2.60547E-06 0.31144 ];
TH232_CAPT                (idx, [1:   4]) = [  9.34234E+19 0.00053  7.42939E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33941E+19 0.00137  1.06514E-01 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  1.45238E+14 0.40486  1.14729E-06 0.40484 ];
SM149_CAPT                (idx, [1:   4]) = [  1.35761E+16 0.04382  1.07914E-04 0.04380 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001235 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15361E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001235 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5166246 5.17173E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3851938 3.85585E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983051 9.83962E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001235 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.01166E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36454E+20 3.7E-06  2.36454E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37765E+19 1.3E-07  9.37765E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25763E+20 0.00032  1.10901E+20 0.00024  1.48617E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19540E+20 0.00018  2.04678E+20 0.00013  1.48617E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43147E+20 0.00036  2.43147E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.35242E+22 0.00025  3.17088E+22 0.00023  1.81537E+21 0.00161 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39256E+19 0.00138 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43465E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09069E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29514E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29514E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.67145E-01 0.06563 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.91518E-01 0.00821 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.75471E-04 0.01504 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.05603E+04 0.02532 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01612E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.67991E-01 0.05483 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.12063E-01 0.05483 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52146E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.72310E-01 0.00042  1.07681E-01 0.00042  3.44765E-04 0.00961 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.72333E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.72511E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.72333E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07843E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02596E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02680E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33310E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33184E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50786E-01 0.00162 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50481E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.31097E-03 0.00606  2.70345E-04 0.02072  8.16857E-04 0.01122  6.08546E-04 0.01385  1.29173E-03 0.00923  2.73266E-04 0.01997  5.02238E-05 0.04400 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40145E-01 0.01366  1.24754E-02 3.1E-05  3.23373E-02 8.7E-05  1.06126E-01 0.00059  2.97441E-01 0.00028  1.23559E+00 0.00022  5.32612E+00 0.03776 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24120E-03 0.00838  2.79230E-04 0.02862  8.15074E-04 0.01633  5.79704E-04 0.01968  1.26343E-03 0.01242  2.58821E-04 0.02788  4.49416E-05 0.06336 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.29768E-01 0.01999  1.24754E-02 4.2E-05  3.23344E-02 0.00013  1.06096E-01 0.00087  2.97327E-01 0.00036  1.23617E+00 0.00029  6.24892E+00 0.03043 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.91525E-07 0.00120  8.91106E-07 0.00118  1.02732E-06 0.04961 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.66800E-07 0.00115  8.66393E-07 0.00113  9.98823E-07 0.04977 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18380E-03 0.00995  2.64582E-04 0.03298  7.92735E-04 0.02013  5.79950E-04 0.02186  1.24027E-03 0.01560  2.65615E-04 0.03112  4.06505E-05 0.08062 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.26594E-01 0.02198  1.24760E-02 4.6E-05  3.23380E-02 0.00015  1.06083E-01 0.00098  2.97716E-01 0.00048  1.23505E+00 0.00041  6.40163E+00 0.04646 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.90476E-07 0.01987  7.90054E-07 0.01987  8.90705E-07 0.05297 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.68803E-07 0.01988  7.68393E-07 0.01988  8.66201E-07 0.05294 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.96529E-03 0.04354  1.96943E-04 0.16486  6.95510E-04 0.07648  4.96989E-04 0.09644  1.22955E-03 0.06146  3.19408E-04 0.12092  2.68957E-05 0.37869 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.31811E-01 0.07884  1.24794E-02 0.0E+00  3.23445E-02 0.00054  1.06572E-01 0.00372  2.96910E-01 0.00147  1.23536E+00 0.00102  6.15892E+00 0.15001 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97022E-03 0.04304  1.94916E-04 0.16065  7.13224E-04 0.07560  5.00813E-04 0.09383  1.21896E-03 0.06037  3.19515E-04 0.11804  2.27922E-05 0.35755 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.22877E-01 0.06874  1.24794E-02 0.0E+00  3.23478E-02 0.00055  1.06566E-01 0.00371  2.96892E-01 0.00145  1.23543E+00 0.00102  6.12632E+00 0.14973 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.76487E+03 0.03921 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.83051E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.58547E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26236E-03 0.00581 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.69537E+03 0.00592 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17499E-09 0.00124 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.17528E-05 0.00424  7.17588E-05 0.00423  1.99021E-06 0.33436 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.64672E-05 0.01497  7.64364E-05 0.01494  2.83987E-06 0.49734 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.52080E-04 0.01456  4.52684E-04 0.01461  2.68058E-04 0.32964 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69757E+01 0.01208 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48084E+01 0.00026  4.03503E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 19:15:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01886E+00  9.96938E-01  9.94322E-01  1.00116E+00  9.99916E-01  1.00260E+00  1.00024E+00  9.98770E-01  9.96619E-01  1.00495E+00  9.98339E-01  9.95845E-01  9.95963E-01  1.00213E+00  9.91348E-01  1.00420E+00  1.00086E+00  9.96947E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.89467E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71053E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31541E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45868E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70960E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50402E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49421E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.49984E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.09832E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333667 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33391E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33391E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.08264E+03 ;
RUNNING_TIME              (idx, 1)        =  1.15934E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.26283E-01  1.25133E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.14468E+02  4.52590E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.73500E-02  1.91000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.21990E+00  2.83710E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.13139E+02  6.26825E+02 ];
CPU_USAGE                 (idx, 1)        = 9.33848 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98975E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.28627E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.39032E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65411E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.25749E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.86422E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.30386E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49479E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49233E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.75624E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39975E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.86101E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.00281E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89501E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.59947E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49186E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34271E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33136E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53982E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.71329E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17586E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.01058E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51364E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.35883E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.51854E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.39209E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61905E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.93154E-04 -1.27632E+26  2.15302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.04590E-01 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.59246E+18 0.00397  1.69966E-02 0.00389 ];
U233_FISS                 (idx, [1:   4]) = [  9.19635E+19 0.00052  9.81598E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.12643E+16 0.04766  1.20250E-04 0.04774 ];
TH232_CAPT                (idx, [1:   4]) = [  9.49957E+19 0.00057  7.37536E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34139E+19 0.00126  1.04146E-01 0.00122 ];
U235_CAPT                 (idx, [1:   4]) = [  4.56355E+15 0.07119  3.53846E-05 0.07120 ];
SM149_CAPT                (idx, [1:   4]) = [  6.25213E+16 0.02202  4.85517E-04 0.02205 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001737 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14361E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001737 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5222031 5.22726E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3798645 3.80224E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981061 9.81937E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001737 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.75089E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36443E+20 3.9E-06  2.36443E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37759E+19 1.2E-07  9.37759E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28757E+20 0.00031  1.13681E+20 0.00022  1.50756E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22533E+20 0.00018  2.07457E+20 0.00012  1.50756E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46403E+20 0.00036  2.46403E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.40300E+22 0.00026  3.21934E+22 0.00024  1.83658E+21 0.00157 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41956E+19 0.00131 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46728E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10859E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29738E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29738E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.40011E-01 0.05111 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.04840E-01 0.01046 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.96737E-04 0.01990 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.82324E+03 0.02965 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01812E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.54704E-01 0.03696 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.80632E-01 0.03696 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52136E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.58510E-01 0.00045  1.06170E-01 0.00044  3.47720E-04 0.00903 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.59429E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.59613E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.59429E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06388E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03522E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03454E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32082E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32159E-02 0.00104 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51118E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51421E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39765E-03 0.00580  2.71131E-04 0.01858  8.48932E-04 0.01094  6.10533E-04 0.01331  1.32015E-03 0.00938  2.87724E-04 0.01805  5.91773E-05 0.04088 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.58334E-01 0.01425  1.24755E-02 2.7E-05  3.23374E-02 9.5E-05  1.06117E-01 0.00063  2.97385E-01 0.00026  1.23558E+00 0.00022  5.34923E+00 0.03569 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28168E-03 0.00844  2.71574E-04 0.02749  8.17233E-04 0.01573  5.86918E-04 0.01895  1.27180E-03 0.01386  2.76538E-04 0.02708  5.76201E-05 0.05853 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56545E-01 0.02063  1.24756E-02 3.8E-05  3.23308E-02 0.00012  1.06088E-01 0.00084  2.97382E-01 0.00040  1.23548E+00 0.00031  6.21377E+00 0.02911 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.07169E-07 0.00130  9.06789E-07 0.00130  1.01784E-06 0.01882 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.69465E-07 0.00117  8.69101E-07 0.00117  9.75664E-07 0.01880 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27323E-03 0.00901  2.60964E-04 0.03279  8.14088E-04 0.01764  5.88172E-04 0.02153  1.27683E-03 0.01563  2.77620E-04 0.03233  5.55520E-05 0.06559 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51796E-01 0.02293  1.24755E-02 5.2E-05  3.23363E-02 0.00016  1.06204E-01 0.00114  2.97569E-01 0.00047  1.23601E+00 0.00041  6.20574E+00 0.04015 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.02618E-07 0.01956  8.02257E-07 0.01956  9.20049E-07 0.05698 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.69362E-07 0.01955  7.69017E-07 0.01955  8.81474E-07 0.05690 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89853E-03 0.04395  2.80781E-04 0.12581  7.67492E-04 0.07541  5.10458E-04 0.08797  1.04537E-03 0.07377  2.47065E-04 0.13551  4.73671E-05 0.31810 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.08628E-01 0.07358  1.24763E-02 0.00012  3.23365E-02 0.00052  1.05760E-01 0.00281  2.97307E-01 0.00162  1.23854E+00 0.00150  5.60108E+00 0.13048 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.86905E-03 0.04293  2.82382E-04 0.12338  7.62575E-04 0.07429  4.99905E-04 0.08644  1.02508E-03 0.07189  2.46664E-04 0.13522  5.24392E-05 0.31012 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.13158E-01 0.07401  1.24764E-02 0.00012  3.23379E-02 0.00052  1.05748E-01 0.00279  2.97242E-01 0.00160  1.23865E+00 0.00150  5.60108E+00 0.13048 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.62141E+03 0.03950 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.01280E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.63828E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.27407E-03 0.00610 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.63378E+03 0.00624 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99417E-09 0.00100 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.98340E-05 0.00562  6.98499E-05 0.00564  1.49234E-06 0.37959 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.96757E-05 0.02058  6.96715E-05 0.02061  2.00092E-06 0.50901 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.84618E-04 0.01965  2.84884E-04 0.01971  2.09346E-04 0.37500 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71215E+01 0.01113 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49421E+01 0.00026  4.05540E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 20:01:37 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01288E+00  9.99189E-01  1.00150E+00  9.90306E-01  1.00340E+00  1.00509E+00  9.94098E-01  9.98738E-01  9.97778E-01  9.99326E-01  1.00362E+00  9.97214E-01  9.96229E-01  1.00281E+00  1.00386E+00  9.92648E-01  9.96881E-01  1.00443E+00  ];
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

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.5E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.88301E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71170E-01 9.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31693E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45967E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70729E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51451E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50469E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.51597E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.09463E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333477 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33389E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33389E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.51717E+03 ;
RUNNING_TIME              (idx, 1)        =  1.62083E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.54800E-01  1.28517E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.60458E+02  4.59903E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.60333E-02  1.86833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.00872E+01  2.86722E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.59257E+02  6.58726E+02 ];
CPU_USAGE                 (idx, 1)        = 9.36046 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99004E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.30915E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.51005E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67024E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.25385E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90409E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33078E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57423E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50535E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.81769E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.56240E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.02352E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.20674E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79305E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74170E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48770E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.35934E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34414E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55697E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.40051E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42525E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08553E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54340E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.27924E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.68461E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.45715E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.16064E-03 -2.49739E+26  2.15424E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.17199E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.60140E+18 0.00379  1.70794E-02 0.00381 ];
U233_FISS                 (idx, [1:   4]) = [  9.18956E+19 0.00051  9.80033E-01 7.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.66111E+16 0.02312  4.97013E-04 0.02309 ];
TH232_CAPT                (idx, [1:   4]) = [  9.59313E+19 0.00055  7.34132E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33745E+19 0.00138  1.02353E-01 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  1.56855E+16 0.04003  1.20023E-04 0.04004 ];
SM149_CAPT                (idx, [1:   4]) = [  1.03233E+17 0.01579  7.90000E-04 0.01580 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001683 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16749E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001683 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5251553 5.25697E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3768629 3.77229E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981501 9.82410E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001683 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.30968E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36434E+20 3.4E-06  2.36434E+20 3.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.2E-07  9.37752E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.30674E+20 0.00032  1.15414E+20 0.00022  1.52600E+19 0.00187 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.24449E+20 0.00019  2.09189E+20 0.00012  1.52600E+19 0.00187 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48572E+20 0.00037  2.48572E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43588E+22 0.00025  3.25036E+22 0.00023  1.85516E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.44207E+19 0.00135 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.48870E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12096E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29702E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29702E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.09094E-01 0.04595 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.18664E-01 0.01124 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.25197E-04 0.02125 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.97220E+03 0.02767 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01762E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.01788E-01 0.03250 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.23000E-01 0.03250 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52128E+00 3.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.50921E-01 0.00043  1.05333E-01 0.00042  3.42069E-04 0.00983 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.51137E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.51206E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.51137E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05475E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04066E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03819E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31371E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31675E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51715E-01 0.00164 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52250E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38755E-03 0.00580  2.78802E-04 0.01889  8.29555E-04 0.01136  6.15118E-04 0.01340  1.33396E-03 0.00897  2.73715E-04 0.01885  5.63955E-05 0.04296 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.54104E-01 0.01426  1.24752E-02 2.9E-05  3.23290E-02 9.5E-05  1.06153E-01 0.00064  2.97596E-01 0.00028  1.23580E+00 0.00024  5.54787E+00 0.03580 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21257E-03 0.00804  2.62531E-04 0.02893  7.87479E-04 0.01705  6.00137E-04 0.01819  1.25219E-03 0.01308  2.53548E-04 0.02712  5.66790E-05 0.06115 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.58592E-01 0.02269  1.24757E-02 3.6E-05  3.23296E-02 0.00013  1.06156E-01 0.00091  2.97576E-01 0.00042  1.23577E+00 0.00030  6.51360E+00 0.02827 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.17246E-07 0.00118  9.16898E-07 0.00118  1.02484E-06 0.01881 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.72190E-07 0.00113  8.71859E-07 0.00114  9.74629E-07 0.01886 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22771E-03 0.01014  2.68795E-04 0.03015  8.07067E-04 0.01861  5.84975E-04 0.02159  1.25840E-03 0.01589  2.62754E-04 0.03145  4.57154E-05 0.07982 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.34299E-01 0.02470  1.24753E-02 5.2E-05  3.23331E-02 0.00016  1.06142E-01 0.00103  2.97646E-01 0.00048  1.23626E+00 0.00042  6.40260E+00 0.04388 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.11176E-07 0.01960  8.10781E-07 0.01960  8.95578E-07 0.07017 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.71365E-07 0.01959  7.70990E-07 0.01960  8.51447E-07 0.07018 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.76490E-03 0.04593  1.94084E-04 0.12859  7.14480E-04 0.08193  5.34339E-04 0.09324  1.04762E-03 0.06970  2.43742E-04 0.12051  3.06363E-05 0.35011 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.58366E-01 0.10068  1.24738E-02 0.00018  3.23363E-02 0.00056  1.06132E-01 0.00325  2.97540E-01 0.00163  1.23650E+00 0.00115  7.25185E+00 0.13122 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.76735E-03 0.04532  1.91067E-04 0.12961  7.18189E-04 0.08119  5.27899E-04 0.09163  1.05547E-03 0.06890  2.44325E-04 0.11706  3.04068E-05 0.35785 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.54013E-01 0.10306  1.24737E-02 0.00018  3.23362E-02 0.00055  1.06135E-01 0.00319  2.97597E-01 0.00162  1.23658E+00 0.00116  7.25185E+00 0.13122 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.41845E+03 0.04169 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.10955E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.66195E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26010E-03 0.00628 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.57940E+03 0.00635 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90043E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84167E-05 0.00626  6.84586E-05 0.00631  1.69363E-06 0.36447 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.18962E-05 0.02341  6.17827E-05 0.02354  1.95171E-06 0.40735 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.17378E-04 0.02026  2.17316E-04 0.02030  2.56414E-04 0.35060 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70182E+01 0.01145 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50469E+01 0.00027  4.06834E+01 0.00037 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 20:49:14 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01434E+00  9.99195E-01  9.94977E-01  9.99450E-01  1.00248E+00  1.00687E+00  9.93635E-01  9.98593E-01  9.94953E-01  1.00127E+00  9.97113E-01  9.96226E-01  1.00755E+00  1.00567E+00  9.95741E-01  9.93213E-01  9.95418E-01  1.00331E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.87627E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71237E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31569E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45817E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69992E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52895E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51912E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.54885E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.09498E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333805 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33370E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33370E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.96438E+03 ;
RUNNING_TIME              (idx, 1)        =  2.09711E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.85517E-01  1.30717E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.07922E+02  4.74639E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.51167E-02  1.90833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.31772E+01  3.08998E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.06661E+02  6.68062E+02 ];
CPU_USAGE                 (idx, 1)        = 9.36710 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98970E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.31591E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.53403E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66405E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.64765E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92590E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34432E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57620E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49763E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00047E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.68135E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.30406E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.30542E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69469E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85070E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93517E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32085E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31458E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51634E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35259E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81327E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08063E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51408E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.56064E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.72744E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.56985E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64039E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.35955E-03 -5.07715E+26  2.15682E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.42674E-01 0.00064 ];
TH232_FISS                (idx, [1:   4]) = [  1.63145E+18 0.00388  1.73907E-02 0.00382 ];
U233_FISS                 (idx, [1:   4]) = [  9.15590E+19 0.00048  9.76027E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.73627E+17 0.01174  1.85115E-03 0.01176 ];
TH232_CAPT                (idx, [1:   4]) = [  9.78059E+19 0.00052  7.30102E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33620E+19 0.00135  9.97429E-02 0.00123 ];
U235_CAPT                 (idx, [1:   4]) = [  6.12200E+16 0.01954  4.56932E-04 0.01950 ];
SM149_CAPT                (idx, [1:   4]) = [  1.45459E+17 0.01211  1.08592E-03 0.01211 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001089 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16904E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001089 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5303213 5.30906E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3713925 3.71772E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983951 9.84908E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001089 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.79397E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36421E+20 3.8E-06  2.36421E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.3E-07  9.37729E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33911E+20 0.00032  1.18408E+20 0.00023  1.55028E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27684E+20 0.00019  2.12181E+20 0.00013  1.55028E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.52328E+20 0.00035  2.52328E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49529E+22 0.00026  3.30717E+22 0.00024  1.88124E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.48527E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.52537E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14157E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29333E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29333E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.55792E-01 0.04708 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07820E-01 0.01182 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.00694E-04 0.02297 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.67868E+03 0.02924 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01513E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.55802E-01 0.03547 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.81266E-01 0.03547 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52121E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.37267E-01 0.00040  1.03812E-01 0.00039  3.34637E-04 0.00977 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.37273E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.36994E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.37273E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03971E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03904E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03949E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31583E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31505E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.55532E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54948E-01 0.00051 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.47960E-03 0.00595  2.86535E-04 0.02020  8.48637E-04 0.01088  6.41982E-04 0.01319  1.35619E-03 0.00920  2.90520E-04 0.01888  5.57362E-05 0.04189 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50639E-01 0.01394  1.24749E-02 3.3E-05  3.23342E-02 0.00010  1.06128E-01 0.00056  2.97648E-01 0.00026  1.23629E+00 0.00026  5.39580E+00 0.03710 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25129E-03 0.00878  2.62835E-04 0.03019  8.02626E-04 0.01665  6.01439E-04 0.01964  1.26171E-03 0.01317  2.66078E-04 0.02776  5.65974E-05 0.06883 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.53779E-01 0.02063  1.24746E-02 4.5E-05  3.23312E-02 0.00014  1.06048E-01 0.00081  2.97498E-01 0.00034  1.23597E+00 0.00033  6.45217E+00 0.02939 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.34459E-07 0.00123  9.34164E-07 0.00124  1.02938E-06 0.01587 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.75800E-07 0.00118  8.75525E-07 0.00119  9.64482E-07 0.01582 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20823E-03 0.00989  2.67810E-04 0.03441  7.79894E-04 0.01829  5.89314E-04 0.02112  1.26089E-03 0.01478  2.57677E-04 0.03230  5.26503E-05 0.07193 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.49888E-01 0.02554  1.24749E-02 6.0E-05  3.23354E-02 0.00016  1.05992E-01 0.00096  2.97521E-01 0.00045  1.23616E+00 0.00049  6.58700E+00 0.04030 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.21777E-07 0.01966  8.21450E-07 0.01966  9.07877E-07 0.05871 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.70304E-07 0.01965  7.69997E-07 0.01965  8.50961E-07 0.05849 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16382E-03 0.04408  2.59544E-04 0.13069  7.62618E-04 0.07860  5.17476E-04 0.09192  1.27742E-03 0.06984  2.81062E-04 0.14234  6.57043E-05 0.29763 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.06366E-01 0.12361  1.24733E-02 0.00017  3.23398E-02 0.00059  1.06289E-01 0.00358  2.98026E-01 0.00183  1.23756E+00 0.00134  6.49591E+00 0.11113 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.16014E-03 0.04320  2.55898E-04 0.12865  7.66079E-04 0.07808  5.19372E-04 0.09109  1.27773E-03 0.06853  2.80741E-04 0.13853  6.03203E-05 0.28090 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.04504E-01 0.11884  1.24733E-02 0.00017  3.23381E-02 0.00059  1.06255E-01 0.00353  2.97997E-01 0.00180  1.23745E+00 0.00131  6.49737E+00 0.11112 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88565E+03 0.04017 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.25689E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.67578E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26701E-03 0.00633 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.52922E+03 0.00628 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89411E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77117E-05 0.00764  6.76915E-05 0.00767  1.59916E-06 0.38382 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.87224E-05 0.02586  5.87976E-05 0.02598  9.86268E-07 0.41376 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.92720E-04 0.02193  1.92689E-04 0.02192  2.02100E-04 0.37530 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75862E+01 0.01219 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51912E+01 0.00027  4.07826E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 21:37:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01661E+00  1.00004E+00  1.00479E+00  1.00035E+00  9.97722E-01  1.00580E+00  9.94895E-01  9.97781E-01  1.00163E+00  9.96257E-01  1.00118E+00  9.98237E-01  9.93283E-01  1.00642E+00  9.94964E-01  9.90911E-01  9.97938E-01  1.00119E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.87300E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71270E-01 8.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31575E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45808E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69719E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54753E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53773E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.58443E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.09751E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334086 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33372E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33372E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.40579E+03 ;
RUNNING_TIME              (idx, 1)        =  2.58067E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.19000E-01  1.33483E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.56108E+02  4.81862E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.52333E-02  2.01167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.77084E+01  4.53112E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.53579E+02  6.82137E+02 ];
CPU_USAGE                 (idx, 1)        = 9.32235 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98908E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26608E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.60789E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67465E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.06958E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96502E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37032E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61038E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50508E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09997E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79418E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.73328E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.48396E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25109E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94548E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34957E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33216E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32288E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52764E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02512E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17095E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.12433E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55048E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.45307E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.83859E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.65788E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96058E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.53946E-03 -7.61602E+26  2.15936E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.64369E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.63043E+18 0.00386  1.73856E-02 0.00382 ];
U233_FISS                 (idx, [1:   4]) = [  9.11449E+19 0.00055  9.71890E-01 8.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.69316E+17 0.00854  3.93837E-03 0.00856 ];
TH232_CAPT                (idx, [1:   4]) = [  9.92520E+19 0.00051  7.25953E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33115E+19 0.00130  9.73644E-02 0.00126 ];
U235_CAPT                 (idx, [1:   4]) = [  1.31389E+17 0.01384  9.61210E-04 0.01388 ];
SM149_CAPT                (idx, [1:   4]) = [  1.66150E+17 0.01143  1.21522E-03 0.01140 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001159 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15426E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001159 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5350376 5.35606E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3670177 3.67390E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 980606 9.81576E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001159 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.45869E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36404E+20 3.8E-06  2.36404E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37698E+19 1.2E-07  9.37698E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36736E+20 0.00030  1.20887E+20 0.00024  1.58492E+19 0.00167 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30506E+20 0.00018  2.14657E+20 0.00014  1.58492E+19 0.00167 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.55263E+20 0.00033  2.55263E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54638E+22 0.00024  3.35446E+22 0.00022  1.91915E+21 0.00162 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50562E+19 0.00125 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.55562E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15957E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28826E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28826E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.75414E-01 0.05024 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.25276E-01 0.01203 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94101E-04 0.02419 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.68728E+03 0.02890 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01846E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.49615E-01 0.03517 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.76058E-01 0.03517 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52111E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.26058E-01 0.00045  1.02573E-01 0.00043  3.39273E-04 0.00906 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.26110E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.26152E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.26110E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02691E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04071E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03859E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31359E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31621E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56364E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.57264E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.56330E-03 0.00585  2.87120E-04 0.01975  8.57201E-04 0.01103  6.56764E-04 0.01294  1.40116E-03 0.00905  2.97570E-04 0.01901  6.34874E-05 0.04322 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.61873E-01 0.01423  1.24756E-02 2.9E-05  3.23251E-02 0.00010  1.06155E-01 0.00060  2.97814E-01 0.00027  1.23724E+00 0.00030  5.60043E+00 0.03454 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28599E-03 0.00792  2.71895E-04 0.02776  7.97350E-04 0.01630  5.92102E-04 0.01983  1.28933E-03 0.01319  2.77495E-04 0.02746  5.78146E-05 0.05890 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.62749E-01 0.01992  1.24757E-02 3.5E-05  3.23236E-02 0.00015  1.06208E-01 0.00088  2.97724E-01 0.00040  1.23739E+00 0.00045  6.50215E+00 0.02710 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.47261E-07 0.00121  9.46909E-07 0.00121  1.05577E-06 0.01567 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.77172E-07 0.00114  8.76846E-07 0.00114  9.77655E-07 0.01567 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.29888E-03 0.00936  2.64540E-04 0.03260  7.82404E-04 0.01917  6.17409E-04 0.02178  1.29427E-03 0.01459  2.80803E-04 0.03276  5.94577E-05 0.06662 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.68312E-01 0.02342  1.24755E-02 6.0E-05  3.23142E-02 0.00017  1.06086E-01 0.00101  2.97775E-01 0.00044  1.23732E+00 0.00056  6.64833E+00 0.03758 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.35351E-07 0.01955  8.34893E-07 0.01955  9.84704E-07 0.05403 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.73784E-07 0.01955  7.73363E-07 0.01955  9.11315E-07 0.05392 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93949E-03 0.04173  2.45663E-04 0.12108  7.45149E-04 0.07958  5.50646E-04 0.09143  1.14415E-03 0.06449  2.00662E-04 0.14654  5.32155E-05 0.27723 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.37066E-01 0.10730  1.24778E-02 8.1E-05  3.23037E-02 0.00061  1.05569E-01 0.00264  2.97984E-01 0.00170  1.23975E+00 0.00202  6.63017E+00 0.11150 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94936E-03 0.04179  2.43694E-04 0.11646  7.60008E-04 0.07820  5.51808E-04 0.09195  1.14040E-03 0.06334  1.99438E-04 0.14168  5.40184E-05 0.26915 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.42616E-01 0.10913  1.24775E-02 9.0E-05  3.23038E-02 0.00061  1.05585E-01 0.00264  2.97880E-01 0.00168  1.24003E+00 0.00206  6.63737E+00 0.11144 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.53300E+03 0.03708 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.40361E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.70778E-07 0.00048 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.31047E-03 0.00570 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.52164E+03 0.00585 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90203E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68068E-05 0.00853  6.68111E-05 0.00854  6.63679E-07 0.58249 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.48396E-05 0.02543  5.48190E-05 0.02537  6.86894E-07 0.81630 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85750E-04 0.02378  1.86113E-04 0.02380  8.64530E-05 0.57559 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74764E+01 0.01080 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53773E+01 0.00026  4.08508E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 22:26:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02055E+00  1.00152E+00  9.95141E-01  1.00599E+00  9.97625E-01  1.00435E+00  9.91986E-01  9.93456E-01  9.97625E-01  1.00518E+00  1.00353E+00  9.94010E-01  9.96626E-01  1.00011E+00  9.96508E-01  9.96900E-01  9.96361E-01  1.00252E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.87239E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71276E-01 9.1E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31123E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45362E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69744E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55776E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54791E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.62066E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10131E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334009 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33418E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33418E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.84121E+03 ;
RUNNING_TIME              (idx, 1)        =  3.07298E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.54733E-01  1.35733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.05165E+02  4.90571E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.13567E-01  1.83333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.38846E+01  6.17618E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.01164E+02  6.87886E+02 ];
CPU_USAGE                 (idx, 1)        = 9.24576 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98986E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.18061E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.64119E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67334E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.14570E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99694E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38982E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61135E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50145E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27586E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.93015E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.77690E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.65814E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93457E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06340E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06578E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30486E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30087E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49799E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.84224E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78259E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13345E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.56227E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.20588E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.89465E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77614E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60097E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.99066E-03 -1.28904E+27  2.16463E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.94357E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.65466E+18 0.00395  1.76409E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  9.02408E+19 0.00051  9.62058E-01 9.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.90539E+17 0.00523  9.49438E-03 0.00524 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00939E+20 0.00051  7.20244E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31759E+19 0.00141  9.40151E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  3.15539E+17 0.00886  2.25162E-03 0.00887 ];
SM149_CAPT                (idx, [1:   4]) = [  1.73781E+17 0.01147  1.24000E-03 0.01146 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002526 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16371E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002526 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5401611 5.40679E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3615618 3.61878E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 985297 9.86064E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002526 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.94298E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36366E+20 4.0E-06  2.36366E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37619E+19 1.2E-07  9.37619E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40106E+20 0.00030  1.24035E+20 0.00021  1.60715E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33868E+20 0.00018  2.17796E+20 0.00012  1.60715E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59205E+20 0.00036  2.59205E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.61018E+22 0.00025  3.41572E+22 0.00022  1.94450E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.55600E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59428E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.18014E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27782E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27782E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.55932E-01 0.04850 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.40350E-01 0.01255 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.87285E-04 0.02397 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.30261E+03 0.02939 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01397E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.55629E-01 0.03369 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.81059E-01 0.03369 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52092E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99703E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.12173E-01 0.00043  1.01029E-01 0.00042  3.33113E-04 0.00991 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12163E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11926E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12163E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01198E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03423E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03225E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32213E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32458E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.61679E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.61984E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.61441E-03 0.00578  2.87600E-04 0.01908  8.68244E-04 0.01234  6.62890E-04 0.01277  1.43808E-03 0.00922  3.00272E-04 0.01870  5.73231E-05 0.04556 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53939E-01 0.01606  1.24755E-02 3.0E-05  3.23218E-02 0.00011  1.06296E-01 0.00056  2.98166E-01 0.00027  1.23961E+00 0.00042  5.16988E+00 0.03936 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29349E-03 0.00835  2.71692E-04 0.02872  8.07900E-04 0.01733  5.94117E-04 0.01831  1.29484E-03 0.01327  2.72751E-04 0.02831  5.21837E-05 0.06978 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48923E-01 0.02349  1.24754E-02 4.2E-05  3.23235E-02 0.00016  1.06218E-01 0.00075  2.98044E-01 0.00037  1.23957E+00 0.00056  6.40392E+00 0.02911 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.62717E-07 0.00116  9.62239E-07 0.00116  1.10836E-06 0.02207 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.78116E-07 0.00108  8.77681E-07 0.00108  1.01078E-06 0.02204 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28189E-03 0.01003  2.53750E-04 0.03337  7.97312E-04 0.01998  6.09975E-04 0.02164  1.29004E-03 0.01545  2.81408E-04 0.03168  4.94010E-05 0.07514 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.54361E-01 0.02471  1.24760E-02 4.9E-05  3.23197E-02 0.00019  1.06184E-01 0.00099  2.97913E-01 0.00047  1.23999E+00 0.00070  6.72233E+00 0.03967 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.49492E-07 0.01962  8.49280E-07 0.01963  9.38649E-07 0.05825 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.75185E-07 0.01963  7.74989E-07 0.01963  8.57106E-07 0.05829 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09514E-03 0.04232  2.30645E-04 0.13587  8.32238E-04 0.07844  5.53813E-04 0.08530  1.19548E-03 0.06341  2.20321E-04 0.14514  6.26461E-05 0.35183 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.51343E-01 0.11109  1.24760E-02 0.00013  3.23050E-02 0.00059  1.05993E-01 0.00294  2.97951E-01 0.00173  1.24182E+00 0.00221  6.99159E+00 0.12193 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.05250E-03 0.04188  2.29387E-04 0.13435  8.21993E-04 0.07580  5.47738E-04 0.08450  1.17434E-03 0.06315  2.18305E-04 0.13919  6.07403E-05 0.34752 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52871E-01 0.11657  1.24764E-02 0.00012  3.23036E-02 0.00059  1.05999E-01 0.00293  2.98021E-01 0.00175  1.24218E+00 0.00227  6.99159E+00 0.12193 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.66979E+03 0.03798 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.56330E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.72288E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28118E-03 0.00660 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.43131E+03 0.00659 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92117E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74829E-05 0.00782  6.75019E-05 0.00783  1.03402E-06 0.45918 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.20247E-05 0.02896  6.18691E-05 0.02913  1.11819E-06 0.67350 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79833E-04 0.02312  1.79982E-04 0.02319  1.36317E-04 0.44502 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69024E+01 0.01248 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54791E+01 0.00027  4.08610E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 15 23:16:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01803E+00  9.98839E-01  9.98976E-01  9.98217E-01  9.90250E-01  1.00115E+00  9.94665E-01  9.94283E-01  1.00136E+00  1.00154E+00  9.96081E-01  1.00718E+00  9.93675E-01  1.00287E+00  1.00545E+00  1.00132E+00  9.92818E-01  1.00330E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.87458E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71254E-01 9.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30839E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45092E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69514E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.56418E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.55433E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.64316E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10492E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333700 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33406E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33406E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.28072E+03 ;
RUNNING_TIME              (idx, 1)        =  3.57317E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.09207E+00  1.37333E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.55005E+02  4.98393E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.33000E-01  1.94333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.04822E+01  6.59748E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.50760E+02  6.95257E+02 ];
CPU_USAGE                 (idx, 1)        = 9.18155 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99048E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.10889E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.72503E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68825E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.20384E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02607E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40450E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.66540E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51430E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.77181E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.23358E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.32177E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.91342E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23096E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33897E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32881E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30488E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29731E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49505E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.56698E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27908E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.17456E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.67760E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.90543E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98538E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.91041E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.21748E-02 -2.61971E+27  2.17794E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02999E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.66201E+18 0.00384  1.77214E-02 0.00383 ];
U233_FISS                 (idx, [1:   4]) = [  8.78765E+19 0.00052  9.36969E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.46574E+18 0.00316  2.62900E-02 0.00310 ];
PU239_FISS                (idx, [1:   4]) = [  1.47044E+15 0.13298  1.56666E-05 0.13303 ];
PU240_FISS                (idx, [1:   4]) = [  2.65283E+13 1.00000  2.81555E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01864E+20 0.00053  7.06483E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28291E+19 0.00138  8.89794E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  8.70513E+17 0.00541  6.03764E-03 0.00540 ];
PU239_CAPT                (idx, [1:   4]) = [  3.41977E+14 0.27175  2.36321E-06 0.27175 ];
PU240_CAPT                (idx, [1:   4]) = [  2.63465E+13 1.00000  1.82239E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80557E+17 0.01126  1.25224E-03 0.01124 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002194 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12990E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002194 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5463024 5.46817E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3553761 3.55692E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 985409 9.86213E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002194 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.16650E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36265E+20 3.9E-06  2.36265E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37387E+19 1.1E-07  9.37387E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44253E+20 0.00028  1.27782E+20 0.00021  1.64707E+19 0.00169 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37991E+20 0.00017  2.21521E+20 0.00012  1.64707E+19 0.00169 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.63680E+20 0.00035  2.63680E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67795E+22 0.00025  3.47962E+22 0.00023  1.98327E+21 0.00163 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60056E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.63997E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20217E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.25135E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.25135E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.98603E-01 0.04951 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.55185E-01 0.01136 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.86230E-04 0.02503 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.31286E+03 0.02944 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01381E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.16177E-01 0.03607 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.45467E-01 0.03607 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52047E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99752E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.96482E-01 0.00044  9.92823E-02 0.00043  3.29919E-04 0.00954 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.95983E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.96062E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.95983E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.94008E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00476E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00297E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36168E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36394E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.71992E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.72106E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.76516E-03 0.00565  2.83741E-04 0.02016  9.04597E-04 0.01077  6.90669E-04 0.01315  1.49955E-03 0.00912  3.23644E-04 0.01901  6.29613E-05 0.04227 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.61294E-01 0.01523  1.24759E-02 2.8E-05  3.23013E-02 0.00011  1.06426E-01 0.00056  2.98975E-01 0.00030  1.24632E+00 0.00054  5.38530E+00 0.03505 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.33393E-03 0.00803  2.62818E-04 0.03012  7.84436E-04 0.01624  6.26536E-04 0.01910  1.31714E-03 0.01322  2.87973E-04 0.02788  5.50223E-05 0.06373 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56934E-01 0.02105  1.24757E-02 3.8E-05  3.23066E-02 0.00017  1.06441E-01 0.00082  2.98848E-01 0.00041  1.24573E+00 0.00075  6.27617E+00 0.02771 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.71569E-07 0.00124  9.71184E-07 0.00125  1.08910E-06 0.01771 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.70938E-07 0.00115  8.70593E-07 0.00116  9.76168E-07 0.01767 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30807E-03 0.00957  2.54129E-04 0.03482  7.75387E-04 0.01826  6.23283E-04 0.02036  1.31696E-03 0.01594  2.81830E-04 0.03106  5.64758E-05 0.06581 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.58364E-01 0.02303  1.24762E-02 5.1E-05  3.22983E-02 0.00020  1.06223E-01 0.00089  2.98994E-01 0.00050  1.24551E+00 0.00101  6.16184E+00 0.03981 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.57268E-07 0.01964  8.56909E-07 0.01964  9.03930E-07 0.05227 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.68453E-07 0.01962  7.68131E-07 0.01962  8.10029E-07 0.05220 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93969E-03 0.04608  1.81233E-04 0.16401  7.30517E-04 0.08096  4.74846E-04 0.09367  1.19951E-03 0.06473  2.83027E-04 0.13690  7.05627E-05 0.26018 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.69160E-01 0.09061  1.24725E-02 0.00021  3.23279E-02 0.00062  1.05590E-01 0.00239  2.97942E-01 0.00160  1.24102E+00 0.00248  5.45143E+00 0.11801 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93829E-03 0.04552  1.86027E-04 0.15618  7.22362E-04 0.07976  4.81140E-04 0.09347  1.19581E-03 0.06358  2.76318E-04 0.13555  7.66263E-05 0.25153 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.75579E-01 0.09221  1.24723E-02 0.00021  3.23278E-02 0.00062  1.05615E-01 0.00240  2.97980E-01 0.00161  1.24111E+00 0.00248  5.45562E+00 0.11782 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.43564E+03 0.04191 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.64969E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.65018E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30892E-03 0.00571 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.43041E+03 0.00587 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93285E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68147E-05 0.00852  6.68008E-05 0.00851  1.21302E-06 0.45503 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.70271E-05 0.02424  5.70460E-05 0.02422  9.61005E-07 0.53890 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.78384E-04 0.02461  1.78552E-04 0.02464  1.31652E-04 0.44547 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65464E+01 0.01131 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.55433E+01 0.00025  4.06495E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 00:07:27 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01616E+00  9.93884E-01  9.97573E-01  9.98156E-01  9.94197E-01  1.00780E+00  9.97975E-01  1.00101E+00  1.00029E+00  9.98176E-01  9.96030E-01  1.00512E+00  9.96309E-01  1.00387E+00  9.93075E-01  9.95829E-01  9.99367E-01  1.00517E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.88487E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71151E-01 9.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30673E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44972E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69811E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54281E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53294E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.60711E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10587E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333831 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33377E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33377E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.72490E+03 ;
RUNNING_TIME              (idx, 1)        =  4.07921E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.23488E+00  1.42817E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.05423E+02  5.04182E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.51567E-01  1.85667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.72048E+01  6.72263E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.01241E+02  7.01245E+02 ];
CPU_USAGE                 (idx, 1)        = 9.13141 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99032E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.05298E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.89930E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73496E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.80064E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03768E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40698E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82721E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55999E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.21375E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78632E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.13244E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.29807E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04639E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84963E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38177E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38167E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35012E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57064E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.85453E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30572E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28140E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.94068E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.47701E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14574E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.95825E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64364E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.29368E-02 -4.93541E+27  2.20110E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.04033E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.62067E+18 0.00370  1.72973E-02 0.00365 ];
U233_FISS                 (idx, [1:   4]) = [  8.43491E+19 0.00055  9.00268E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  4.87790E+18 0.00243  5.20621E-02 0.00236 ];
PU239_FISS                (idx, [1:   4]) = [  2.58831E+16 0.03369  2.76347E-04 0.03371 ];
PU240_FISS                (idx, [1:   4]) = [  7.94948E+14 0.17975  8.51133E-06 0.17983 ];
PU241_FISS                (idx, [1:   4]) = [  8.20558E+14 0.17627  8.75586E-06 0.17621 ];
TH232_CAPT                (idx, [1:   4]) = [  9.96729E+19 0.00052  6.84170E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23065E+19 0.00147  8.44747E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  1.72591E+18 0.00357  1.18466E-02 0.00352 ];
U238_CAPT                 (idx, [1:   4]) = [  2.11916E+14 0.34940  1.45536E-06 0.34940 ];
PU239_CAPT                (idx, [1:   4]) = [  1.29212E+16 0.04374  8.86500E-05 0.04366 ];
PU240_CAPT                (idx, [1:   4]) = [  2.38898E+15 0.11526  1.63807E-05 0.11512 ];
PU241_CAPT                (idx, [1:   4]) = [  1.58974E+14 0.40484  1.09036E-06 0.40486 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85586E+17 0.01278  1.27382E-03 0.01277 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001297 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12528E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001297 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486246 5.49186E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3528502 3.53194E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 986549 9.87449E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001297 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.58908E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36145E+20 4.5E-06  2.36145E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36994E+19 1.2E-07  9.36994E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45598E+20 0.00029  1.29126E+20 0.00021  1.64720E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39297E+20 0.00018  2.22825E+20 0.00012  1.64720E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65275E+20 0.00033  2.65275E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68988E+22 0.00024  3.49150E+22 0.00022  1.98380E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61952E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65493E+20 0.00018 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20376E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.18334E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.18334E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.19430E-01 0.05092 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.22775E-01 0.01319 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.91651E-04 0.02552 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.56587E+03 0.02857 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01258E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.91553E-01 0.03786 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.23212E-01 0.03786 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52024E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99836E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.90255E-01 0.00044  9.85654E-02 0.00043  3.37625E-04 0.00903 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90465E-01 0.00018 ];
COL_KEFF                  (idx, [1:   2]) = [  8.90217E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90465E-01 0.00018 ];
ABS_KINF                  (idx, [1:   2]) = [  9.88063E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94357E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94159E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44768E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.45030E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.85888E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86168E-01 0.00051 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94882E-03 0.00547  2.97750E-04 0.01886  9.44024E-04 0.01072  7.01998E-04 0.01256  1.58616E-03 0.00838  3.45454E-04 0.01850  7.34377E-05 0.03606 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.82866E-01 0.01413  1.24754E-02 2.9E-05  3.22605E-02 0.00012  1.06667E-01 0.00056  3.00087E-01 0.00030  1.25456E+00 0.00065  6.05778E+00 0.02914 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44909E-03 0.00799  2.59786E-04 0.02918  8.24278E-04 0.01557  6.12292E-04 0.01797  1.38891E-03 0.01278  2.98553E-04 0.02626  6.52718E-05 0.05479 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.83333E-01 0.02034  1.24760E-02 3.4E-05  3.22615E-02 0.00017  1.06654E-01 0.00084  3.00181E-01 0.00044  1.25448E+00 0.00083  6.78372E+00 0.02327 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.53331E-07 0.00142  9.52694E-07 0.00142  1.14743E-06 0.01972 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.48644E-07 0.00131  8.48078E-07 0.00131  1.02150E-06 0.01973 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41711E-03 0.00899  2.54959E-04 0.03344  8.20740E-04 0.01830  6.21749E-04 0.02139  1.36403E-03 0.01464  2.88876E-04 0.03365  6.67605E-05 0.06624 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.86249E-01 0.02562  1.24756E-02 4.8E-05  3.22607E-02 0.00020  1.06674E-01 0.00099  2.99951E-01 0.00051  1.25129E+00 0.00110  6.72707E+00 0.03308 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.44631E-07 0.01968  8.44167E-07 0.01969  9.25825E-07 0.05215 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.52167E-07 0.01969  7.51752E-07 0.01969  8.24606E-07 0.05218 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11024E-03 0.04332  2.15960E-04 0.13427  7.72745E-04 0.07462  5.09368E-04 0.10341  1.28990E-03 0.06148  2.30234E-04 0.15487  9.20382E-05 0.22412 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.34376E-01 0.11812  1.24773E-02 0.00012  3.23132E-02 0.00080  1.06360E-01 0.00306  2.99806E-01 0.00185  1.24649E+00 0.00327  7.24492E+00 0.08059 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09599E-03 0.04278  2.16162E-04 0.13369  7.76503E-04 0.07620  5.04386E-04 0.10176  1.28755E-03 0.05903  2.23612E-04 0.15447  8.77849E-05 0.22580 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.29417E-01 0.11699  1.24772E-02 0.00012  3.23115E-02 0.00079  1.06375E-01 0.00301  2.99806E-01 0.00184  1.24644E+00 0.00325  7.24465E+00 0.08059 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.70195E+03 0.03866 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.46318E-07 0.00082 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.42412E-07 0.00067 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46254E-03 0.00623 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.65964E+03 0.00627 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90083E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68303E-05 0.00699  6.67846E-05 0.00703  1.62047E-06 0.38519 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.77636E-05 0.02790  5.76914E-05 0.02789  1.60901E-06 0.53186 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.82294E-04 0.02513  1.82312E-04 0.02517  1.71625E-04 0.37460 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63311E+01 0.01099 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53294E+01 0.00027  4.00823E+01 0.00038 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 00:58:40 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01671E+00  1.00655E+00  1.00395E+00  1.00078E+00  9.94423E-01  9.96163E-01  9.98965E-01  9.91787E-01  1.00306E+00  9.97544E-01  9.97069E-01  9.96589E-01  9.97956E-01  1.00209E+00  1.00240E+00  9.99593E-01  9.96800E-01  9.97564E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.88859E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71114E-01 9.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30572E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44891E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69751E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53259E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52274E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59085E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10548E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334018 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33419E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33419E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.17543E+03 ;
RUNNING_TIME              (idx, 1)        =  4.59132E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.38232E+00  1.47433E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.56439E+02  5.10162E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.70117E-01  1.85500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.38925E+01  6.68757E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.52491E+02  7.05411E+02 ];
CPU_USAGE                 (idx, 1)        = 9.09417 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99037E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.01142E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.92513E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74191E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.21460E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03070E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40171E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85980E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56726E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.20233E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99960E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.02760E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.48577E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13520E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04323E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81801E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38546E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35172E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57348E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90415E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73573E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29245E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99333E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.91148E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.15901E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.95547E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30374E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.81589E-02 -6.05908E+27  2.21233E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03705E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.60016E+18 0.00434  1.70697E-02 0.00422 ];
U233_FISS                 (idx, [1:   4]) = [  8.33309E+19 0.00059  8.89017E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  5.56220E+18 0.00211  5.93406E-02 0.00204 ];
U238_FISS                 (idx, [1:   4]) = [  2.65167E+13 1.00000  2.82990E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  5.73369E+16 0.02038  6.11699E-04 0.02038 ];
PU240_FISS                (idx, [1:   4]) = [  2.33556E+15 0.11042  2.48744E-05 0.11040 ];
PU241_FISS                (idx, [1:   4]) = [  2.62282E+15 0.11069  2.80132E-05 0.11097 ];
TH232_CAPT                (idx, [1:   4]) = [  9.84361E+19 0.00054  6.76126E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21511E+19 0.00151  8.34630E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  1.96571E+18 0.00360  1.35019E-02 0.00357 ];
U238_CAPT                 (idx, [1:   4]) = [  3.98753E+14 0.26909  2.74129E-06 0.26958 ];
PU239_CAPT                (idx, [1:   4]) = [  2.73259E+16 0.02979  1.87713E-04 0.02981 ];
PU240_CAPT                (idx, [1:   4]) = [  5.86275E+15 0.06601  4.02887E-05 0.06613 ];
PU241_CAPT                (idx, [1:   4]) = [  5.04269E+14 0.22242  3.46053E-06 0.22242 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83155E+17 0.01260  1.25823E-03 0.01262 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002583 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10929E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002583 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5485321 5.49016E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3531747 3.53470E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 985515 9.86238E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002583 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.24797E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36139E+20 4.3E-06  2.36139E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36849E+19 1.1E-07  9.36849E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45606E+20 0.00028  1.29154E+20 0.00019  1.64524E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39291E+20 0.00017  2.22839E+20 0.00011  1.64524E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65182E+20 0.00036  2.65182E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68369E+22 0.00025  3.48592E+22 0.00022  1.97766E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61543E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65445E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20064E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.14847E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.14847E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.59474E-01 0.04976 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31622E-01 0.01176 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93636E-04 0.02420 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.40615E+03 0.02807 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01379E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.18254E-01 0.03547 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.47468E-01 0.03547 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52057E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99867E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.90980E-01 0.00045  9.86545E-02 0.00046  3.41679E-04 0.00946 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90594E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.90512E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90594E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.88050E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92047E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.91926E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.48152E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.48304E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90703E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90829E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.97842E-03 0.00564  2.81637E-04 0.01967  9.47768E-04 0.01089  7.17029E-04 0.01284  1.59647E-03 0.00843  3.60181E-04 0.01707  7.53403E-05 0.03963 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.95442E-01 0.01434  1.24760E-02 3.7E-05  3.22526E-02 0.00012  1.06587E-01 0.00054  3.00634E-01 0.00026  1.25667E+00 0.00072  6.33648E+00 0.02826 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48692E-03 0.00775  2.50941E-04 0.02795  8.25185E-04 0.01555  6.21548E-04 0.01849  1.40401E-03 0.01161  3.22417E-04 0.02465  6.28102E-05 0.06012 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.95616E-01 0.02075  1.24765E-02 3.4E-05  3.22593E-02 0.00018  1.06595E-01 0.00077  3.00563E-01 0.00039  1.25687E+00 0.00097  7.23880E+00 0.02135 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.39032E-07 0.00130  9.38498E-07 0.00130  1.09488E-06 0.01633 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.36609E-07 0.00123  8.36134E-07 0.00123  9.75410E-07 0.01632 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45334E-03 0.00959  2.42139E-04 0.03317  8.19184E-04 0.01862  6.06913E-04 0.02158  1.39409E-03 0.01487  3.21846E-04 0.02848  6.91676E-05 0.06246 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.12950E-01 0.02486  1.24754E-02 6.2E-05  3.22496E-02 0.00023  1.06599E-01 0.00111  3.00652E-01 0.00054  1.25743E+00 0.00129  7.24652E+00 0.02933 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.28029E-07 0.01972  8.27284E-07 0.01971  1.00058E-06 0.06435 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.37937E-07 0.01971  7.37274E-07 0.01971  8.91150E-07 0.06405 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09889E-03 0.04425  2.22564E-04 0.15671  7.79239E-04 0.07676  5.31266E-04 0.09658  1.22815E-03 0.06995  2.95885E-04 0.11604  4.17889E-05 0.30049 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.88879E-01 0.12678  1.24775E-02 0.00013  3.22238E-02 0.00063  1.06760E-01 0.00337  3.00693E-01 0.00192  1.25550E+00 0.00398  5.63362E+00 0.12823 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09300E-03 0.04430  2.22278E-04 0.15417  7.72790E-04 0.07472  5.22749E-04 0.09737  1.24309E-03 0.06949  2.91057E-04 0.11618  4.10361E-05 0.29821 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.81105E-01 0.12590  1.24775E-02 0.00013  3.22228E-02 0.00061  1.06784E-01 0.00339  3.00668E-01 0.00189  1.25543E+00 0.00405  5.63362E+00 0.12823 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.76604E+03 0.04045 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.32760E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.31012E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46092E-03 0.00561 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.71020E+03 0.00552 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90309E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75005E-05 0.00848  6.75054E-05 0.00850  1.51852E-06 0.39017 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.63355E-05 0.02646  5.64325E-05 0.02646  1.01381E-06 0.42948 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.86523E-04 0.02314  1.86570E-04 0.02322  1.72522E-04 0.37563 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60396E+01 0.01052 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52274E+01 0.00026  3.98762E+01 0.00038 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 01:50:33 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01576E+00  9.96682E-01  1.00046E+00  1.00502E+00  9.94800E-01  1.00114E+00  1.00026E+00  1.00018E+00  1.00181E+00  9.96334E-01  9.98641E-01  1.00035E+00  9.95452E-01  1.00487E+00  9.98137E-01  9.97588E-01  9.93267E-01  9.99234E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.89053E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71095E-01 9.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30622E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44945E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69688E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52808E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51824E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.58023E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.10549E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334130 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33416E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33416E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.63170E+03 ;
RUNNING_TIME              (idx, 1)        =  5.11021E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.52873E+00  1.46417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.08131E+02  5.16920E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.89717E-01  1.96000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.06888E+01  6.79632E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.04266E+02  7.08995E+02 ];
CPU_USAGE                 (idx, 1)        = 9.06362 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99045E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.97741E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.92566E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74244E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.77471E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01673E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39211E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87435E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56880E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.31557E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.18165E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.06105E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.66601E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21209E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20668E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20265E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37743E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34451E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56420E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92748E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11781E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28541E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01477E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.26037E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14060E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.95476E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96383E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.34359E-02 -7.19454E+27  2.22369E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03591E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.61071E+18 0.00398  1.71973E-02 0.00396 ];
U233_FISS                 (idx, [1:   4]) = [  8.24344E+19 0.00055  8.80130E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  6.02173E+18 0.00211  6.42915E-02 0.00202 ];
U238_FISS                 (idx, [1:   4]) = [  5.28889E+13 0.70603  5.68598E-07 0.70601 ];
PU239_FISS                (idx, [1:   4]) = [  1.05897E+17 0.01659  1.13041E-03 0.01654 ];
PU240_FISS                (idx, [1:   4]) = [  4.77452E+15 0.07639  5.09721E-05 0.07649 ];
PU241_FISS                (idx, [1:   4]) = [  5.85428E+15 0.07169  6.24849E-05 0.07166 ];
TH232_CAPT                (idx, [1:   4]) = [  9.74531E+19 0.00058  6.68940E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20191E+19 0.00154  8.25037E-02 0.00151 ];
U235_CAPT                 (idx, [1:   4]) = [  2.13094E+18 0.00340  1.46273E-02 0.00336 ];
U238_CAPT                 (idx, [1:   4]) = [  5.56725E+14 0.21080  3.82063E-06 0.21081 ];
PU239_CAPT                (idx, [1:   4]) = [  4.74700E+16 0.02358  3.25906E-04 0.02359 ];
PU240_CAPT                (idx, [1:   4]) = [  1.39746E+16 0.04313  9.58789E-05 0.04309 ];
PU241_CAPT                (idx, [1:   4]) = [  1.30040E+15 0.15164  8.91856E-06 0.15138 ];
SM149_CAPT                (idx, [1:   4]) = [  1.88097E+17 0.01189  1.29121E-03 0.01190 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002467 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11916E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002467 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5489281 5.49419E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3529300 3.53232E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983886 9.84675E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002467 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.77534E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36158E+20 4.5E-06  2.36158E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36721E+19 1.1E-07  9.36721E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45707E+20 0.00030  1.29206E+20 0.00020  1.65013E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39380E+20 0.00018  2.22878E+20 0.00012  1.65013E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65159E+20 0.00036  2.65159E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68139E+22 0.00025  3.48293E+22 0.00022  1.98456E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61103E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65490E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19934E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.11418E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.11418E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.99413E-01 0.04793 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.22149E-01 0.01378 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93287E-04 0.02411 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.14557E+03 0.02920 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01536E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.27639E-01 0.03458 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.56063E-01 0.03458 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52111E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99894E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.90468E-01 0.00045  9.86056E-02 0.00045  3.41953E-04 0.00892 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90508E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.90662E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90508E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87756E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90163E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90302E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50969E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50731E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.95060E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.94631E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.98841E-03 0.00585  2.86675E-04 0.02149  9.36914E-04 0.01136  7.40276E-04 0.01312  1.59233E-03 0.00864  3.55036E-04 0.01942  7.71810E-05 0.03948 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.97448E-01 0.01510  1.24347E-02 0.00334  3.22464E-02 0.00013  1.06620E-01 0.00055  3.00766E-01 0.00032  1.25914E+00 0.00082  6.27880E+00 0.02756 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48644E-03 0.00854  2.43505E-04 0.03185  8.15154E-04 0.01702  6.56621E-04 0.01865  1.39312E-03 0.01246  3.11058E-04 0.02631  6.69801E-05 0.06128 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.93209E-01 0.02237  1.24758E-02 5.7E-05  3.22429E-02 0.00017  1.06782E-01 0.00082  3.00761E-01 0.00047  1.25970E+00 0.00106  6.87078E+00 0.02267 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.35259E-07 0.00137  9.34741E-07 0.00136  1.08788E-06 0.01586 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32769E-07 0.00129  8.32308E-07 0.00129  9.68640E-07 0.01585 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44824E-03 0.00891  2.43377E-04 0.03572  8.22788E-04 0.01893  6.49807E-04 0.02047  1.35712E-03 0.01493  3.11165E-04 0.03288  6.39871E-05 0.06442 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.89619E-01 0.02401  1.24760E-02 6.1E-05  3.22426E-02 0.00024  1.06666E-01 0.00101  3.00592E-01 0.00054  1.25646E+00 0.00149  6.98234E+00 0.03154 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.27576E-07 0.01970  8.27424E-07 0.01970  8.43656E-07 0.05395 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.37091E-07 0.01969  7.36955E-07 0.01969  7.51648E-07 0.05398 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85224E-03 0.04320  1.76093E-04 0.15579  6.02879E-04 0.08264  5.64643E-04 0.09124  1.23988E-03 0.06501  2.22080E-04 0.15326  4.66747E-05 0.28413 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.62747E-01 0.09335  1.24782E-02 0.00013  3.22405E-02 0.00075  1.06156E-01 0.00282  3.00611E-01 0.00198  1.26319E+00 0.00474  7.75472E+00 0.07925 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85889E-03 0.04204  1.76139E-04 0.14655  6.00742E-04 0.07991  5.72278E-04 0.08998  1.23876E-03 0.06353  2.18316E-04 0.14894  5.26513E-05 0.27820 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.73590E-01 0.09266  1.24782E-02 0.00013  3.22417E-02 0.00074  1.06143E-01 0.00276  3.00554E-01 0.00197  1.26321E+00 0.00473  7.75472E+00 0.07925 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.47707E+03 0.03907 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.27122E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.25519E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.36199E-03 0.00607 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.62744E+03 0.00620 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90083E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.64219E-05 0.00727  6.64087E-05 0.00729  1.15206E-06 0.44539 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.74254E-05 0.02511  5.73369E-05 0.02497  1.74755E-06 0.63487 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84187E-04 0.02387  1.84421E-04 0.02377  1.31864E-04 0.44699 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61769E+01 0.01105 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51824E+01 0.00027  3.97056E+01 0.00039 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 02:43:12 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01249E+00  9.98866E-01  9.98954E-01  9.92966E-01  9.95103E-01  9.95798E-01  9.98082E-01  9.99224E-01  1.00249E+00  9.99969E-01  9.98984E-01  9.96970E-01  9.99390E-01  1.00787E+00  9.94691E-01  1.00021E+00  1.00142E+00  1.00653E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.91355E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70865E-01 9.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30202E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44628E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69779E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51420E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50437E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.56592E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.11568E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333944 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33384E+04 0.00069 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33384E+04 0.00069 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.09215E+03 ;
RUNNING_TIME              (idx, 1)        =  5.63672E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.67698E+00  1.48250E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.60581E+02  5.24495E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.09567E-01  1.98500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.78634E+01  7.17455E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.56540E+02  7.12120E+02 ];
CPU_USAGE                 (idx, 1)        = 9.03389 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99048E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.94430E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.94310E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74771E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.07214E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97935E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36984E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92916E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57630E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.80122E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75883E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.31106E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04624E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44237E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70316E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34083E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37171E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33748E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55613E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.96899E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27092E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27864E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.06463E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.42032E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11095E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.97096E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60422E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.44673E-02 -1.17200E+28  2.26894E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03302E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.56216E+18 0.00421  1.66890E-02 0.00413 ];
U233_FISS                 (idx, [1:   4]) = [  8.05025E+19 0.00058  8.60075E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  6.89812E+18 0.00218  7.36947E-02 0.00204 ];
U238_FISS                 (idx, [1:   4]) = [  1.06002E+14 0.49751  1.13562E-06 0.49754 ];
PU239_FISS                (idx, [1:   4]) = [  3.27243E+17 0.00896  3.49630E-03 0.00895 ];
PU240_FISS                (idx, [1:   4]) = [  2.63473E+16 0.03278  2.81444E-04 0.03282 ];
PU241_FISS                (idx, [1:   4]) = [  3.89845E+16 0.02393  4.16512E-04 0.02392 ];
TH232_CAPT                (idx, [1:   4]) = [  9.53315E+19 0.00056  6.51929E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17147E+19 0.00149  8.01116E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.44626E+18 0.00329  1.67293E-02 0.00328 ];
U238_CAPT                 (idx, [1:   4]) = [  1.35465E+15 0.13947  9.25963E-06 0.13940 ];
PU239_CAPT                (idx, [1:   4]) = [  1.53677E+17 0.01306  1.05095E-03 0.01305 ];
PU240_CAPT                (idx, [1:   4]) = [  6.71716E+16 0.01930  4.59300E-04 0.01929 ];
PU241_CAPT                (idx, [1:   4]) = [  8.20602E+15 0.05637  5.61351E-05 0.05638 ];
SM149_CAPT                (idx, [1:   4]) = [  1.89011E+17 0.01205  1.29260E-03 0.01204 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001513 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08507E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001513 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498238 5.50362E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3519640 3.52281E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983635 9.84416E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001513 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.58794E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36329E+20 4.5E-06  2.36329E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36326E+19 9.7E-08  9.36326E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46131E+20 0.00029  1.29545E+20 0.00021  1.65860E+19 0.00173 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39764E+20 0.00018  2.23178E+20 0.00012  1.65860E+19 0.00173 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65699E+20 0.00036  2.65699E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68991E+22 0.00024  3.49061E+22 0.00022  1.99301E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61569E+19 0.00157 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65921E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19805E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.98438E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.98438E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.53947E-01 0.04442 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33631E-01 0.01026 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.32412E-04 0.02188 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.95487E+03 0.02941 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01562E-01 0.00016 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.72542E-01 0.03042 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.96389E-01 0.03042 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52400E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99979E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.89220E-01 0.00049  9.84570E-02 0.00048  3.38686E-04 0.00793 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.89704E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.89496E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.89704E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86879E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.86129E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.86061E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.57184E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.57261E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.03944E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.04157E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.98794E-03 0.00516  3.00551E-04 0.02007  9.07834E-04 0.01138  7.30810E-04 0.01269  1.60391E-03 0.00847  3.61790E-04 0.01835  8.30502E-05 0.04031 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.02971E-01 0.01573  1.24772E-02 4.9E-05  3.22165E-02 0.00014  1.06735E-01 0.00058  3.01136E-01 0.00032  1.25987E+00 0.00087  5.97509E+00 0.02910 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46499E-03 0.00748  2.59335E-04 0.02983  7.90277E-04 0.01670  6.31824E-04 0.01828  1.39773E-03 0.01294  3.13706E-04 0.02437  7.21202E-05 0.05636 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00396E-01 0.02149  1.24779E-02 7.1E-05  3.22108E-02 0.00020  1.06666E-01 0.00077  3.01259E-01 0.00045  1.26038E+00 0.00118  6.74190E+00 0.02226 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.18839E-07 0.00132  9.18365E-07 0.00132  1.05709E-06 0.01771 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16995E-07 0.00124  8.16574E-07 0.00124  9.40008E-07 0.01776 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41625E-03 0.00793  2.53513E-04 0.03313  7.74925E-04 0.01941  6.27051E-04 0.02122  1.38363E-03 0.01408  3.07950E-04 0.02946  6.91808E-05 0.06277 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08724E-01 0.02494  1.24760E-02 5.9E-05  3.22146E-02 0.00025  1.06771E-01 0.00103  3.01228E-01 0.00055  1.25690E+00 0.00174  7.14375E+00 0.03027 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.08885E-07 0.01974  8.08528E-07 0.01974  9.16236E-07 0.06435 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.19276E-07 0.01972  7.18958E-07 0.01972  8.15180E-07 0.06445 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91550E-03 0.04291  2.00357E-04 0.16970  5.70734E-04 0.09127  5.92336E-04 0.10136  1.21630E-03 0.05708  2.81246E-04 0.13328  5.45302E-05 0.24123 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.14056E-01 0.09197  1.24766E-02 0.00014  3.22295E-02 0.00082  1.06789E-01 0.00337  3.01547E-01 0.00200  1.25697E+00 0.00575  7.47746E+00 0.08299 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90604E-03 0.04236  2.02647E-04 0.16472  5.84699E-04 0.08885  5.85879E-04 0.10062  1.18565E-03 0.05596  2.89334E-04 0.13170  5.78335E-05 0.24160 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.16040E-01 0.09208  1.24766E-02 0.00014  3.22283E-02 0.00082  1.06831E-01 0.00338  3.01478E-01 0.00197  1.25712E+00 0.00574  7.48177E+00 0.08301 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.62556E+03 0.03863 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.11238E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10232E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.45027E-03 0.00533 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.78720E+03 0.00542 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93700E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79558E-05 0.00684  6.79879E-05 0.00711  2.14944E-06 0.34600 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65532E-05 0.02245  5.65345E-05 0.02252  1.84521E-06 0.43253 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.20233E-04 0.02154  2.20210E-04 0.02156  2.32366E-04 0.32953 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63006E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50437E+01 0.00026  3.93067E+01 0.00038 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 03:36:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00762E+00  9.95058E-01  9.98439E-01  9.98958E-01  9.93422E-01  1.00630E+00  1.00231E+00  1.00172E+00  1.00486E+00  9.97312E-01  9.96954E-01  1.00180E+00  1.00169E+00  1.00208E+00  9.92437E-01  9.99556E-01  9.93916E-01  1.00556E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.96498E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70350E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.28596E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.43266E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71549E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51119E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50122E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61173E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.14814E+00 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334120 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33383E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33383E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.55426E+03 ;
RUNNING_TIME              (idx, 1)        =  6.16586E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.82753E+00  1.50550E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.13289E+02  5.27079E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.28417E-01  1.88500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.51400E+01  7.27655E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.09351E+02  7.14589E+02 ];
CPU_USAGE                 (idx, 1)        = 9.00809 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99045E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.91561E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.92421E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74198E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.97753E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93042E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34332E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95923E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57327E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.49337E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38541E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.31866E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13555E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69263E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23911E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57890E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34674E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31751E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52933E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00370E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59825E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24146E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04227E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.11355E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.03396E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.99287E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32052E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.09159E-01 -2.34882E+28  2.38663E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02722E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.53799E+18 0.00419  1.64486E-02 0.00413 ];
U233_FISS                 (idx, [1:   4]) = [  7.92185E+19 0.00054  8.47271E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.10800E+18 0.00195  7.60226E-02 0.00189 ];
U238_FISS                 (idx, [1:   4]) = [  1.06947E+14 0.49750  1.15028E-06 0.49750 ];
PU239_FISS                (idx, [1:   4]) = [  6.20570E+17 0.00677  6.63724E-03 0.00675 ];
PU240_FISS                (idx, [1:   4]) = [  7.45424E+16 0.01851  7.97364E-04 0.01851 ];
PU241_FISS                (idx, [1:   4]) = [  1.30697E+17 0.01345  1.39808E-03 0.01348 ];
TH232_CAPT                (idx, [1:   4]) = [  9.33986E+19 0.00052  6.36963E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15387E+19 0.00156  7.86926E-02 0.00152 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51093E+18 0.00325  1.71244E-02 0.00324 ];
U238_CAPT                 (idx, [1:   4]) = [  2.84439E+15 0.09679  1.94381E-05 0.09696 ];
PU239_CAPT                (idx, [1:   4]) = [  2.91609E+17 0.00889  1.98891E-03 0.00891 ];
PU240_CAPT                (idx, [1:   4]) = [  1.94317E+17 0.01247  1.32507E-03 0.01243 ];
PU241_CAPT                (idx, [1:   4]) = [  2.88545E+16 0.03055  1.96822E-04 0.03058 ];
SM149_CAPT                (idx, [1:   4]) = [  1.95331E+17 0.01165  1.33198E-03 0.01162 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001488 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09835E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001488 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498126 5.50360E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3506219 3.50936E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 997143 9.98027E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001488 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.24100E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36623E+20 5.0E-06  2.36623E+20 5.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35945E+19 1.1E-07  9.35945E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46719E+20 0.00030  1.29592E+20 0.00020  1.71273E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40314E+20 0.00019  2.23187E+20 0.00011  1.71273E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66429E+20 0.00034  2.66429E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.72415E+22 0.00025  3.52186E+22 0.00022  2.02286E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.65912E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66905E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20052E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.73230E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.73230E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.10580E-01 0.03985 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24926E-01 0.00942 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.95181E-04 0.02041 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.46166E+03 0.03006 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.00203E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.44148E-01 0.02366 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.59880E-01 0.02366 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52817E+00 5.0E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00060E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.87213E-01 0.00045  9.82333E-02 0.00044  3.45420E-04 0.00841 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.87513E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88158E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.87513E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.85833E-01 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83372E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83505E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61576E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61333E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10605E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10496E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03666E-03 0.00459  2.89975E-04 0.01979  9.30293E-04 0.01096  7.23237E-04 0.01302  1.63003E-03 0.00758  3.78560E-04 0.01790  8.45607E-05 0.03737 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.15382E-01 0.01462  1.24795E-02 9.4E-05  3.22009E-02 0.00015  1.06821E-01 0.00060  3.01531E-01 0.00032  1.26005E+00 0.00096  6.39425E+00 0.02565 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51303E-03 0.00707  2.46286E-04 0.02909  8.07785E-04 0.01589  6.20803E-04 0.01809  1.42104E-03 0.01213  3.43298E-04 0.02614  7.38230E-05 0.05234 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.21792E-01 0.02183  1.24805E-02 0.00015  3.21960E-02 0.00022  1.06843E-01 0.00078  3.01714E-01 0.00044  1.26121E+00 0.00125  6.99046E+00 0.02083 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.18705E-07 0.00139  9.18119E-07 0.00139  1.08101E-06 0.02037 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.15034E-07 0.00131  8.14514E-07 0.00131  9.58930E-07 0.02033 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.51299E-03 0.00854  2.56002E-04 0.03483  8.11178E-04 0.01913  6.32129E-04 0.02216  1.40756E-03 0.01417  3.33437E-04 0.02988  7.26845E-05 0.06549 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.16510E-01 0.02624  1.24796E-02 0.00016  3.22017E-02 0.00027  1.06751E-01 0.00107  3.01603E-01 0.00056  1.25635E+00 0.00224  7.04089E+00 0.03025 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.04745E-07 0.01977  8.04576E-07 0.01978  8.24563E-07 0.05038 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.13984E-07 0.01977  7.13833E-07 0.01977  7.31717E-07 0.05044 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.46162E-03 0.04334  3.08313E-04 0.12159  8.36692E-04 0.08364  6.50937E-04 0.09420  1.30715E-03 0.06365  2.92072E-04 0.12162  6.64598E-05 0.24838 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.89905E-01 0.09329  1.24713E-02 0.00021  3.22195E-02 0.00081  1.06273E-01 0.00253  3.02058E-01 0.00205  1.26318E+00 0.00383  7.08632E+00 0.08455 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.43145E-03 0.04299  3.03270E-04 0.12243  8.38456E-04 0.08413  6.29267E-04 0.09330  1.30649E-03 0.06140  2.85502E-04 0.11763  6.84624E-05 0.24736 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.96924E-01 0.09394  1.24714E-02 0.00021  3.22194E-02 0.00082  1.06283E-01 0.00254  3.02007E-01 0.00202  1.26350E+00 0.00386  7.08632E+00 0.08455 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.32529E+03 0.03919 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.09501E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.06877E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.62868E-03 0.00582 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.99074E+03 0.00592 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.10453E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.81158E-05 0.00542  6.81144E-05 0.00541  9.76811E-07 0.50622 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.86265E-05 0.01906  5.86508E-05 0.01904  5.80438E-07 0.59416 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.83202E-04 0.02003  2.83950E-04 0.02002  1.01382E-04 0.49984 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61444E+01 0.01198 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50122E+01 0.00028  3.90351E+01 0.00039 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 04:29:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02098E+00  9.99968E-01  9.98351E-01  9.97875E-01  9.92525E-01  1.00003E+00  9.97792E-01  9.93911E-01  9.99012E-01  1.00040E+00  1.00744E+00  9.99982E-01  1.00086E+00  9.98841E-01  9.97532E-01  1.00350E+00  9.92466E-01  9.98532E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.02475E-02 0.00037  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69753E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.27326E-01 0.00020  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.42275E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72699E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51635E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50627E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.65945E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.18437E+00 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334149 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33401E+04 0.00069 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33401E+04 0.00069 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.01730E+03 ;
RUNNING_TIME              (idx, 1)        =  6.69612E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.97988E+00  1.52350E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.66105E+02  5.28161E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.46633E-01  1.82167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.24357E+01  7.29560E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.62357E+02  7.15235E+02 ];
CPU_USAGE                 (idx, 1)        = 8.98625 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99060E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.89133E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.92000E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74034E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.38799E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91696E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33586E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96838E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57226E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57516E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55639E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39288E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14800E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76524E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39704E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93170E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34039E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31287E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52282E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01926E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00959E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23029E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03207E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27563E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01172E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  8.00972E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98062E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.64991E-01 -3.55018E+28  2.50676E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02082E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.53365E+18 0.00397  1.63823E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  7.92023E+19 0.00060  8.46008E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.07977E+18 0.00183  7.56242E-02 0.00177 ];
U238_FISS                 (idx, [1:   4]) = [  2.13472E+14 0.34942  2.28381E-06 0.34943 ];
PU239_FISS                (idx, [1:   4]) = [  6.56852E+17 0.00628  7.01681E-03 0.00631 ];
PU240_FISS                (idx, [1:   4]) = [  8.55741E+16 0.01744  9.13939E-04 0.01742 ];
PU241_FISS                (idx, [1:   4]) = [  1.55797E+17 0.01287  1.66406E-03 0.01284 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27305E+19 0.00053  6.32092E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15087E+19 0.00147  7.84479E-02 0.00139 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48589E+18 0.00321  1.69455E-02 0.00322 ];
U238_CAPT                 (idx, [1:   4]) = [  2.90578E+15 0.08973  1.97866E-05 0.08970 ];
PU239_CAPT                (idx, [1:   4]) = [  3.10513E+17 0.00837  2.11640E-03 0.00832 ];
PU240_CAPT                (idx, [1:   4]) = [  2.24271E+17 0.01130  1.52875E-03 0.01130 ];
PU241_CAPT                (idx, [1:   4]) = [  3.13212E+16 0.02952  2.13516E-04 0.02953 ];
SM149_CAPT                (idx, [1:   4]) = [  1.98556E+17 0.01160  1.35356E-03 0.01161 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002029 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07126E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002029 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5489899 5.49480E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3503526 3.50647E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1008604 1.00945E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002029 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.45058E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36680E+20 5.0E-06  2.36680E+20 5.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35894E+19 1.1E-07  9.35894E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46671E+20 0.00028  1.28998E+20 0.00019  1.76722E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40260E+20 0.00017  2.22588E+20 0.00011  1.76722E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66991E+20 0.00035  2.66991E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.76079E+22 0.00025  3.55498E+22 0.00023  2.05801E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.69528E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67213E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20436E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.52054E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.52054E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.21937E-01 0.03197 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21840E-01 0.00819 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.86699E-04 0.01602 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.09228E+03 0.03295 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99062E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 9.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.07541E-01 0.01706 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.15965E-01 0.01706 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52892E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.86728E-01 0.00049  9.81904E-02 0.00047  3.40870E-04 0.00805 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.86698E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.86506E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.86698E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86284E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83238E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83247E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61791E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61754E-02 0.00103 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11819E-01 0.00139 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11643E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.02368E-03 0.00500  2.92149E-04 0.01932  9.38556E-04 0.01098  7.16081E-04 0.01257  1.60982E-03 0.00883  3.84598E-04 0.01749  8.24804E-05 0.03675 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.07144E-01 0.01419  1.24807E-02 0.00014  3.21963E-02 0.00015  1.06783E-01 0.00055  3.01461E-01 0.00031  1.25852E+00 0.00104  6.24504E+00 0.02565 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46338E-03 0.00810  2.41112E-04 0.02903  8.02243E-04 0.01648  6.23397E-04 0.01715  1.39575E-03 0.01289  3.30916E-04 0.02528  6.99608E-05 0.05210 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.04912E-01 0.01917  1.24828E-02 0.00024  3.21942E-02 0.00023  1.06802E-01 0.00077  3.01463E-01 0.00041  1.25805E+00 0.00123  6.73724E+00 0.02242 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.30181E-07 0.00152  9.29789E-07 0.00153  1.04493E-06 0.01736 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.24757E-07 0.00144  8.24410E-07 0.00144  9.26358E-07 0.01731 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45935E-03 0.00828  2.47703E-04 0.03709  8.15743E-04 0.01911  6.04891E-04 0.02156  1.39511E-03 0.01348  3.26564E-04 0.03089  6.93425E-05 0.06594 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06378E-01 0.02498  1.24805E-02 0.00016  3.21971E-02 0.00030  1.06684E-01 0.00092  3.01504E-01 0.00058  1.25977E+00 0.00177  6.82189E+00 0.03160 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.16015E-07 0.01988  8.15507E-07 0.01988  9.19015E-07 0.05018 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23724E-07 0.01988  7.23272E-07 0.01988  8.15671E-07 0.05022 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.22288E-03 0.04499  2.80408E-04 0.14802  6.83858E-04 0.09674  6.10482E-04 0.09023  1.31973E-03 0.06404  2.92133E-04 0.14400  3.62746E-05 0.30169 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.48669E-01 0.07682  1.24893E-02 0.00084  3.22089E-02 0.00086  1.06743E-01 0.00315  3.01532E-01 0.00203  1.25678E+00 0.00572  7.15185E+00 0.09778 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.22735E-03 0.04407  2.79355E-04 0.14568  6.72959E-04 0.09458  6.12673E-04 0.08894  1.33514E-03 0.06275  2.93630E-04 0.14526  3.35933E-05 0.29277 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.49139E-01 0.07696  1.24893E-02 0.00084  3.22092E-02 0.00084  1.06662E-01 0.00305  3.01497E-01 0.00202  1.25719E+00 0.00574  7.14511E+00 0.09771 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.99234E+03 0.04108 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.20007E-07 0.00086 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.15740E-07 0.00072 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47562E-03 0.00610 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77877E+03 0.00619 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.30894E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92891E-05 0.00458  6.92985E-05 0.00459  1.95587E-06 0.35188 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.17581E-05 0.01821  6.18334E-05 0.01824  1.04887E-06 0.41663 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.67769E-04 0.01569  3.68347E-04 0.01570  2.30580E-04 0.32942 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62549E+01 0.01204 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50627E+01 0.00028  3.90042E+01 0.00037 ];


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
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha5' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 15 17:19:32 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 16 04:57:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650053972373 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02446E+00  9.95842E-01  1.00582E+00  9.94676E-01  9.93235E-01  1.00334E+00  9.94739E-01  9.98591E-01  1.00550E+00  1.00214E+00  1.00691E+00  9.86408E-01  9.97631E-01  1.00596E+00  9.88187E-01  1.00022E+00  9.98356E-01  9.97979E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.08736E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69126E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.25994E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.41233E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73863E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52503E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51487E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.71625E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.22374E+00 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334160 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33398E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33398E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.26643E+03 ;
RUNNING_TIME              (idx, 1)        =  6.98073E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.87200E-01  9.87200E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.13305E+00  1.53167E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.94352E+02  2.82475E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.65250E-01  1.86167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.62091E+01  3.77340E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.94342E+02  6.94342E+02 ];
CPU_USAGE                 (idx, 1)        = 8.97676 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98398E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.88132E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22641.52;
MEMSIZE                   (idx, 1)        = 22437.16;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 285.45;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 204.36;

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

TOT_ACTIVITY              (idx, 1)        =  6.91367E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73967E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.55754E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90691E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32950E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.97208E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57220E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.58342E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60183E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39889E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14595E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78665E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44436E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03339E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33916E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31211E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52163E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02572E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13669E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22274E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03001E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.29697E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99256E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  8.02024E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64071E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.21310E-01 -4.76203E+28  2.62795E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01514E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.53564E+18 0.00411  1.64036E-02 0.00406 ];
U233_FISS                 (idx, [1:   4]) = [  7.92005E+19 0.00052  8.46027E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.07681E+18 0.00179  7.55925E-02 0.00166 ];
U238_FISS                 (idx, [1:   4]) = [  4.00774E+14 0.26858  4.27301E-06 0.26868 ];
PU239_FISS                (idx, [1:   4]) = [  6.61417E+17 0.00639  7.06465E-03 0.00633 ];
PU240_FISS                (idx, [1:   4]) = [  8.78373E+16 0.01727  9.38316E-04 0.01725 ];
PU241_FISS                (idx, [1:   4]) = [  1.56648E+17 0.01360  1.67338E-03 0.01360 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21654E+19 0.00056  6.27706E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15215E+19 0.00147  7.84696E-02 0.00142 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48029E+18 0.00339  1.68923E-02 0.00336 ];
U238_CAPT                 (idx, [1:   4]) = [  3.23496E+15 0.08545  2.20494E-05 0.08541 ];
PU239_CAPT                (idx, [1:   4]) = [  3.10639E+17 0.00913  2.11592E-03 0.00917 ];
PU240_CAPT                (idx, [1:   4]) = [  2.26252E+17 0.01095  1.54093E-03 0.01094 ];
PU241_CAPT                (idx, [1:   4]) = [  3.46319E+16 0.02722  2.35843E-04 0.02720 ];
SM149_CAPT                (idx, [1:   4]) = [  1.98078E+17 0.01276  1.34914E-03 0.01277 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001952 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07346E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001952 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487164 5.49219E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3498812 3.50174E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1015976 1.01681E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001952 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.11759E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36687E+20 4.5E-06  2.36687E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35892E+19 1.1E-07  9.35892E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46769E+20 0.00026  1.28524E+20 0.00018  1.82440E+19 0.00151 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40358E+20 0.00016  2.22114E+20 0.00010  1.82440E+19 0.00151 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.67341E+20 0.00034  2.67341E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.79683E+22 0.00024  3.58800E+22 0.00022  2.08825E+21 0.00152 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.71842E+19 0.00128 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67542E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20825E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.34432E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.34432E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.31161E-01 0.03020 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.01259E-01 0.00771 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.87517E-04 0.01543 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.17768E+03 0.03597 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.98328E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.43194E-01 0.01232 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.47293E-01 0.01232 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52900E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.85524E-01 0.00046  9.80631E-02 0.00045  3.36959E-04 0.00898 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.85632E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.85368E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.85632E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.85912E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83387E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83544E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61543E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61267E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.12012E-01 0.00134 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11763E-01 0.00042 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.01034E-03 0.00514  2.85560E-04 0.02074  9.32983E-04 0.01055  7.18181E-04 0.01325  1.61769E-03 0.00790  3.72625E-04 0.01753  8.33017E-05 0.03728 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10635E-01 0.01421  1.24778E-02 7.5E-05  3.21939E-02 0.00017  1.06852E-01 0.00056  3.01507E-01 0.00029  1.25606E+00 0.00115  6.37308E+00 0.02569 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44773E-03 0.00699  2.42360E-04 0.02748  7.83066E-04 0.01627  6.29333E-04 0.01851  1.41836E-03 0.01200  3.10617E-04 0.02476  6.39948E-05 0.04839 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.94902E-01 0.01826  1.24779E-02 6.5E-05  3.21839E-02 0.00023  1.06820E-01 0.00077  3.01386E-01 0.00044  1.25546E+00 0.00136  6.87926E+00 0.02156 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.42841E-07 0.00170  9.42481E-07 0.00171  1.04992E-06 0.01610 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.34859E-07 0.00164  8.34539E-07 0.00165  9.29718E-07 0.01612 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41867E-03 0.00907  2.53888E-04 0.03248  8.06626E-04 0.01838  6.01446E-04 0.02289  1.37630E-03 0.01438  3.16242E-04 0.02938  6.41647E-05 0.06530 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.92200E-01 0.02361  1.24776E-02 8.8E-05  3.21990E-02 0.00030  1.06814E-01 0.00102  3.01644E-01 0.00053  1.26048E+00 0.00211  6.78565E+00 0.03186 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.19027E-07 0.01970  8.18522E-07 0.01970  9.80450E-07 0.05672 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.25668E-07 0.01971  7.25221E-07 0.01971  8.67838E-07 0.05651 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09600E-03 0.04341  2.72361E-04 0.14888  7.80428E-04 0.08238  5.06439E-04 0.10505  1.16440E-03 0.06423  3.28755E-04 0.12347  4.36234E-05 0.32046 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.66509E-01 0.07919  1.24972E-02 0.00123  3.21921E-02 0.00101  1.06514E-01 0.00330  3.00134E-01 0.00192  1.27443E+00 0.00478  6.03812E+00 0.11324 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10948E-03 0.04290  2.68698E-04 0.14663  7.77444E-04 0.08041  5.24402E-04 0.10111  1.18188E-03 0.06389  3.17682E-04 0.12186  3.93743E-05 0.30857 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.60421E-01 0.07635  1.24964E-02 0.00119  3.21936E-02 0.00100  1.06520E-01 0.00330  3.00199E-01 0.00193  1.27302E+00 0.00499  6.03421E+00 0.11376 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.80705E+03 0.03931 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.29419E-07 0.00084 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.22967E-07 0.00068 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.41749E-03 0.00555 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.67777E+03 0.00561 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.55101E-09 0.00111 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.96735E-05 0.00400  6.96657E-05 0.00401  3.81024E-06 0.25455 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.26032E-05 0.01418  6.25456E-05 0.01412  4.65680E-06 0.36178 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.65102E-04 0.01493  4.65370E-04 0.01490  4.05065E-04 0.24470 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60217E+01 0.01100 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51487E+01 0.00025  3.90353E+01 0.00036 ];

