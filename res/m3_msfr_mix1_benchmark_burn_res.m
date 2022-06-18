
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 15:20:45 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03335E+00  9.95768E-01  9.97871E-01  9.93463E-01  9.92797E-01  1.00602E+00  1.00675E+00  1.00300E+00  9.81179E-01  1.01432E+00  9.84307E-01  1.00389E+00  1.00095E+00  9.92216E-01  1.00072E+00  1.00567E+00  9.90975E-01  9.96754E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.32453E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66755E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39517E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53403E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62970E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46934E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46027E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16071E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.68206E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500360 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50057E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50057E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.02777E+02 ;
RUNNING_TIME              (idx, 1)        =  2.13426E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.88217E-01  1.88217E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.01180E+01  2.01180E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.00100E-01  4.86500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.13378E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.50105 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97674E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.49595E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  2.80028E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.20219E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13449E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.80028E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.20219E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.69141E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.42870E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.69141E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.42870E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29717E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.13351E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.31722E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.46146E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.68527E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.54826E+18 0.00405  1.65082E-02 0.00398 ];
U233_FISS                 (idx, [1:   4]) = [  9.22326E+19 0.00047  9.83492E-01 6.7E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.17500E+19 0.00059  7.54847E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34069E+19 0.00130  1.10307E-01 0.00129 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002297 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13644E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002297 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5133889 5.13865E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3961269 3.96482E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907139 9.07899E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002297 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.24100E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36439E+20 3.8E-06  2.36439E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.2E-07  9.37766E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21571E+20 0.00032  1.08272E+20 0.00024  1.32989E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15348E+20 0.00018  2.02049E+20 0.00013  1.32989E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36536E+20 0.00038  2.36536E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.13047E+22 0.00028  2.96867E+22 0.00025  1.61801E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.14761E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.36824E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05614E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.18268E-01 0.08524 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.96076E-01 0.00823 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.20629E-04 0.01388 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.65627E+03 0.02315 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09220E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.73767E-01 0.06975 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.39727E-01 0.06975 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52130E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.99568E-01 0.00042  6.22813E-02 0.00040  1.97344E-04 0.00872 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.99542E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.99644E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.99542E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09935E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04888E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04875E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30310E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30300E-02 0.00101 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47306E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46589E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.20098E-03 0.00525  2.57176E-04 0.01856  7.93119E-04 0.01118  5.87791E-04 0.01418  1.24628E-03 0.00868  2.68292E-04 0.01910  4.83161E-05 0.04574 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45479E-01 0.01568  1.24760E-02 2.6E-05  3.23288E-02 8.9E-05  1.06002E-01 0.00056  2.97547E-01 0.00028  1.23539E+00 0.00022  4.39525E+00 0.04293 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19625E-03 0.00801  2.59221E-04 0.02786  7.92890E-04 0.01702  5.84017E-04 0.02032  1.24702E-03 0.01286  2.67535E-04 0.02769  4.55677E-05 0.07111 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.39611E-01 0.02244  1.24753E-02 4.1E-05  3.23323E-02 0.00014  1.05982E-01 0.00079  2.97387E-01 0.00037  1.23530E+00 0.00029  6.38350E+00 0.02899 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.36362E-07 0.00119  8.36041E-07 0.00119  9.35017E-07 0.01826 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.35946E-07 0.00112  8.35626E-07 0.00113  9.34437E-07 0.01825 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.15611E-03 0.00888  2.48587E-04 0.03215  7.67897E-04 0.01823  5.82097E-04 0.02140  1.24796E-03 0.01427  2.63655E-04 0.03130  4.59115E-05 0.07536 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.42004E-01 0.02384  1.24756E-02 4.8E-05  3.23261E-02 0.00015  1.06058E-01 0.00105  2.97344E-01 0.00047  1.23446E+00 0.00042  6.38779E+00 0.04218 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.43398E-07 0.01716  7.43135E-07 0.01716  7.72932E-07 0.04980 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.43207E-07 0.01715  7.42943E-07 0.01715  7.72841E-07 0.04974 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80612E-03 0.04328  1.82399E-04 0.14480  7.06343E-04 0.08161  4.97012E-04 0.09269  1.13003E-03 0.06473  2.47005E-04 0.14135  4.33270E-05 0.35571 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.38864E-01 0.10207  1.24750E-02 0.00015  3.23162E-02 0.00040  1.06509E-01 0.00343  2.96630E-01 0.00135  1.23454E+00 0.00117  4.58998E+00 0.15223 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.82626E-03 0.04281  1.92560E-04 0.13786  7.10521E-04 0.08001  5.03724E-04 0.09221  1.13174E-03 0.06453  2.41944E-04 0.13884  4.57736E-05 0.33178 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.36837E-01 0.10186  1.24750E-02 0.00015  3.23133E-02 0.00039  1.06527E-01 0.00343  2.96722E-01 0.00136  1.23458E+00 0.00117  4.58998E+00 0.15223 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81298E+03 0.04067 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.31736E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.31310E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.13013E-03 0.00527 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.76446E+03 0.00535 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.27738E-09 0.00135 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.24580E-05 0.00410  7.24630E-05 0.00410  2.82999E-06 0.25490 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.82078E-05 0.01558  7.82242E-05 0.01564  2.73411E-06 0.31546 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.01145E-04 0.01381  5.01045E-04 0.01387  5.52698E-04 0.24996 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72416E+01 0.01044 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46027E+01 0.00028  4.04528E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 15:52:55 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01774E+00  1.00902E+00  9.95239E-01  1.01035E+00  9.98446E-01  1.01070E+00  9.84548E-01  9.93352E-01  9.98100E-01  1.00304E+00  9.88173E-01  1.00206E+00  9.90321E-01  1.00019E+00  9.97578E-01  9.98355E-01  9.91295E-01  1.01149E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.31738E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66826E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39465E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53326E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63281E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47618E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46709E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17595E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.68086E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500556 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50053E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50053E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.22780E+02 ;
RUNNING_TIME              (idx, 1)        =  5.35038E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.12067E-01  1.23850E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.21309E+01  3.20130E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.80500E-02  1.80500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.50300E-01  5.01500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.34993E+01  3.16877E+02 ];
CPU_USAGE                 (idx, 1)        = 9.77089 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98338E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77789E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  5.76422E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60318E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65774E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.46880E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.03467E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26475E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46896E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.44818E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65317E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.38502E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.75025E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06315E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17815E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44280E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35676E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17529E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55425E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59319E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38168E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.38937E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51285E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50937E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.28081E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.55500E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08103E-04 -2.38689E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.79476E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.54733E+18 0.00416  1.65020E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  9.21772E+19 0.00050  9.83197E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.58416E+14 0.27070  3.80998E-06 0.27091 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28131E+19 0.00053  7.50538E-01 0.00025 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34241E+19 0.00126  1.08555E-01 0.00118 ];
U235_CAPT                 (idx, [1:   4]) = [  1.66767E+14 0.37512  1.35627E-06 0.37513 ];
SM149_CAPT                (idx, [1:   4]) = [  1.27760E+16 0.04392  1.03342E-04 0.04393 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002114 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.17266E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002114 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5171697 5.17688E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3921178 3.92481E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909239 9.10034E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002114 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36432E+20 3.8E-06  2.36432E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37763E+19 1.3E-07  9.37763E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23734E+20 0.00030  1.10289E+20 0.00022  1.34448E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17510E+20 0.00017  2.04065E+20 0.00012  1.34448E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38875E+20 0.00038  2.38875E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.16512E+22 0.00026  3.00153E+22 0.00024  1.63591E+21 0.00187 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.17396E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39250E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06825E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.70652E-01 0.06826 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.89100E-01 0.00875 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.21212E-04 0.01666 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.45827E+03 0.02151 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09005E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 9.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.46424E-01 0.06006 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.05854E-01 0.06006 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52123E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.89653E-01 0.00044  6.16456E-02 0.00042  2.00377E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.89384E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.89828E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.89384E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08839E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05063E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05047E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30070E-02 0.00126 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30074E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47132E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.47050E-01 0.00051 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.27817E-03 0.00580  2.64238E-04 0.01916  8.03082E-04 0.01064  5.99645E-04 0.01352  1.29341E-03 0.00890  2.66373E-04 0.01997  5.14137E-05 0.04607 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43814E-01 0.01519  1.24446E-02 0.00251  3.23391E-02 8.8E-05  1.06077E-01 0.00058  2.97199E-01 0.00026  1.23267E+00 0.00252  4.34786E+00 0.04238 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23233E-03 0.00891  2.60822E-04 0.03089  7.93669E-04 0.01573  5.79175E-04 0.02049  1.28430E-03 0.01345  2.63976E-04 0.03025  5.03802E-05 0.06769 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45716E-01 0.02123  1.24756E-02 4.0E-05  3.23392E-02 0.00013  1.06176E-01 0.00086  2.97133E-01 0.00037  1.23598E+00 0.00029  6.18489E+00 0.02928 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.45653E-07 0.00117  8.45391E-07 0.00117  9.22575E-07 0.01588 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.36842E-07 0.00109  8.36582E-07 0.00109  9.13039E-07 0.01588 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23926E-03 0.00894  2.68992E-04 0.03116  7.99455E-04 0.01722  5.88703E-04 0.02073  1.27293E-03 0.01444  2.58901E-04 0.03116  5.02806E-05 0.07154 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.41312E-01 0.02442  1.24758E-02 4.3E-05  3.23394E-02 0.00015  1.06018E-01 0.00097  2.97080E-01 0.00041  1.23644E+00 0.00036  6.32856E+00 0.04033 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.50864E-07 0.01711  7.50531E-07 0.01710  8.03891E-07 0.05033 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.43235E-07 0.01710  7.42906E-07 0.01710  7.95360E-07 0.05032 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98570E-03 0.04051  2.45631E-04 0.13614  6.84416E-04 0.07602  5.80756E-04 0.08575  1.22200E-03 0.06104  2.19117E-04 0.13952  3.37762E-05 0.28987 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.60947E-01 0.11693  1.24759E-02 0.00013  3.23298E-02 0.00048  1.06703E-01 0.00351  2.97831E-01 0.00163  1.23292E+00 0.00118  8.03379E+00 0.09455 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99793E-03 0.03932  2.49651E-04 0.13515  6.88637E-04 0.07403  5.73082E-04 0.08344  1.23230E-03 0.05982  2.21588E-04 0.13417  3.26718E-05 0.29377 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.56923E-01 0.11400  1.24759E-02 0.00013  3.23307E-02 0.00049  1.06704E-01 0.00351  2.97780E-01 0.00161  1.23293E+00 0.00118  8.03379E+00 0.09455 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.98919E+03 0.03682 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.39729E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.30977E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26288E-03 0.00529 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.88548E+03 0.00523 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16187E-09 0.00127 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.15502E-05 0.00451  7.15244E-05 0.00451  1.95935E-06 0.32464 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.36538E-05 0.01680  7.36734E-05 0.01680  1.55156E-06 0.42725 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.06919E-04 0.01595  4.07146E-04 0.01593  3.25285E-04 0.32454 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72603E+01 0.01140 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46709E+01 0.00028  4.04967E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 16:27:01 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01514E+00  1.00491E+00  9.97931E-01  9.98930E-01  1.00479E+00  1.00678E+00  9.90542E-01  1.00049E+00  9.88119E-01  9.99806E-01  9.91300E-01  9.98467E-01  9.95801E-01  9.99878E-01  1.00999E+00  1.00743E+00  9.94286E-01  9.95403E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.29829E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67017E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39507E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53300E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62777E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48475E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47565E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19255E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.67181E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499974 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50033E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50033E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.62155E+02 ;
RUNNING_TIME              (idx, 1)        =  8.76074E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.43150E-01  1.31083E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  8.60762E+01  3.39452E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.60667E-02  1.80167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.00033E-01  4.96833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.76028E+01  4.73548E+02 ];
CPU_USAGE                 (idx, 1)        = 9.84113 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98408E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.85132E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.38061E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65348E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.23588E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85278E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29500E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49687E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49293E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.74704E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39665E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.84987E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.95765E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89696E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60088E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49545E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34591E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33320E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54274E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.66427E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17885E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.00405E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51062E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.32787E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.49737E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.67876E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.75785E-04 -1.27132E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.96439E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.55963E+18 0.00383  1.66339E-02 0.00379 ];
U233_FISS                 (idx, [1:   4]) = [  9.20772E+19 0.00052  9.82031E-01 6.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.11711E+16 0.04474  1.19037E-04 0.04470 ];
TH232_CAPT                (idx, [1:   4]) = [  9.42519E+19 0.00054  7.45343E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34150E+19 0.00126  1.06087E-01 0.00119 ];
U235_CAPT                 (idx, [1:   4]) = [  3.77267E+15 0.08442  2.98557E-05 0.08439 ];
SM149_CAPT                (idx, [1:   4]) = [  6.12088E+16 0.01934  4.84018E-04 0.01934 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001302 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14918E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001302 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5220618 5.22609E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3871109 3.87500E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909575 9.10409E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001302 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.19792E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36422E+20 3.8E-06  2.36422E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37758E+19 1.2E-07  9.37758E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26456E+20 0.00029  1.12867E+20 0.00023  1.35888E+19 0.00173 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20232E+20 0.00017  2.06643E+20 0.00012  1.35888E+19 0.00173 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41969E+20 0.00037  2.41969E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.20888E+22 0.00026  3.04392E+22 0.00024  1.64967E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20301E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42262E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08415E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51431E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51431E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.59654E-01 0.05423 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13037E-01 0.01075 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.75235E-04 0.01930 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.70411E+03 0.02211 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08964E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.55828E-01 0.04003 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.96153E-01 0.04003 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52114E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77000E-01 0.00044  6.08647E-02 0.00043  1.96528E-04 0.00949 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.77040E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.77128E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.77040E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07489E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05546E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05757E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29449E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29155E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48617E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48065E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.31503E-03 0.00608  2.65913E-04 0.01935  8.04188E-04 0.01133  6.08741E-04 0.01314  1.31672E-03 0.00957  2.69337E-04 0.02109  5.01253E-05 0.04476 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42276E-01 0.01537  1.24760E-02 2.8E-05  3.23318E-02 9.4E-05  1.06045E-01 0.00056  2.97524E-01 0.00029  1.23238E+00 0.00252  4.45269E+00 0.04259 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24967E-03 0.00856  2.63093E-04 0.02896  7.90767E-04 0.01641  5.89068E-04 0.01932  1.29631E-03 0.01452  2.62737E-04 0.02809  4.76920E-05 0.06616 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.39391E-01 0.02173  1.24760E-02 3.5E-05  3.23317E-02 0.00013  1.06080E-01 0.00082  2.97485E-01 0.00042  1.23554E+00 0.00032  6.21644E+00 0.02999 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.59533E-07 0.00127  8.59173E-07 0.00128  9.67419E-07 0.01559 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39699E-07 0.00119  8.39347E-07 0.00120  9.45101E-07 0.01559 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21061E-03 0.00972  2.55186E-04 0.03290  7.71321E-04 0.01952  5.87541E-04 0.02235  1.27293E-03 0.01484  2.72492E-04 0.03133  5.11438E-05 0.06846 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51264E-01 0.02352  1.24763E-02 4.9E-05  3.23281E-02 0.00015  1.06030E-01 0.00101  2.97455E-01 0.00046  1.23496E+00 0.00042  6.33904E+00 0.04003 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.61365E-07 0.01713  7.60790E-07 0.01713  9.18571E-07 0.05773 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.43888E-07 0.01712  7.43328E-07 0.01712  8.97055E-07 0.05761 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.66932E-03 0.04694  1.95868E-04 0.15918  6.64034E-04 0.08106  4.98837E-04 0.09007  1.04332E-03 0.07243  2.20398E-04 0.14024  4.68671E-05 0.29665 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.47974E-01 0.12004  1.24763E-02 0.00014  3.23055E-02 0.00035  1.05772E-01 0.00292  2.97019E-01 0.00152  1.23236E+00 0.00137  6.55270E+00 0.13203 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.65134E-03 0.04593  1.89599E-04 0.16031  6.65048E-04 0.07845  5.07075E-04 0.08885  1.02260E-03 0.06995  2.19213E-04 0.13858  4.78012E-05 0.31433 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.46212E-01 0.12179  1.24763E-02 0.00014  3.23071E-02 0.00036  1.05777E-01 0.00291  2.97026E-01 0.00151  1.23233E+00 0.00136  6.55270E+00 0.13203 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.51513E+03 0.04413 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.53881E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.34178E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17173E-03 0.00602 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.71477E+03 0.00602 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.00252E-09 0.00095 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.94621E-05 0.00571  6.94476E-05 0.00573  1.74536E-06 0.33642 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.96229E-05 0.02229  6.97452E-05 0.02233  6.88145E-07 0.46642 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.65577E-04 0.01869  2.65559E-04 0.01861  2.80815E-04 0.33260 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71332E+01 0.01208 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47565E+01 0.00025  4.06345E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 17:01:38 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01552E+00  1.00712E+00  9.96267E-01  1.00382E+00  9.97867E-01  1.00638E+00  9.84211E-01  9.97443E-01  1.00229E+00  1.00359E+00  9.94824E-01  9.97560E-01  9.98024E-01  1.00071E+00  9.94922E-01  9.99238E-01  9.96580E-01  1.00363E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.28575E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67143E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39765E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53507E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62064E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49936E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49030E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.21193E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66849E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2501147 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50061E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50061E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.20665E+03 ;
RUNNING_TIME              (idx, 1)        =  1.22229E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.79533E-01  1.36383E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.20531E+02  3.44545E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.46667E-02  1.86000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.52383E-01  5.22833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.22224E+02  4.97327E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87207 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98422E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88361E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.49723E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66931E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.15102E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88967E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31987E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57622E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50592E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.80033E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.55798E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.00408E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.14877E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79516E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74308E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48837E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36254E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34598E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55993E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39277E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42581E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07667E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54003E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.15332E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.65712E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.76121E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12696E-03 -2.48830E+26  2.21046E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.09835E-01 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.57817E+18 0.00377  1.68386E-02 0.00368 ];
U233_FISS                 (idx, [1:   4]) = [  9.18855E+19 0.00054  9.80452E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.20427E+16 0.02370  4.48589E-04 0.02366 ];
TH232_CAPT                (idx, [1:   4]) = [  9.52373E+19 0.00056  7.41424E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34425E+19 0.00134  1.04654E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  1.41096E+16 0.03771  1.09871E-04 0.03771 ];
SM149_CAPT                (idx, [1:   4]) = [  9.89406E+16 0.01515  7.70145E-04 0.01511 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002446 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15659E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002446 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5258796 5.26380E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3837079 3.84044E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906571 9.07331E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002446 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.94298E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36415E+20 3.9E-06  2.36415E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37750E+19 1.3E-07  9.37750E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28453E+20 0.00031  1.14607E+20 0.00022  1.38468E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22228E+20 0.00018  2.08382E+20 0.00012  1.38468E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.44030E+20 0.00037  2.44030E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24088E+22 0.00026  3.07326E+22 0.00023  1.67616E+21 0.00168 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.21428E+19 0.00158 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44371E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09696E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51439E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51439E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.75218E-01 0.05013 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.09668E-01 0.01179 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.30529E-04 0.02089 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.50510E+03 0.02283 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09271E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.70645E-01 0.03837 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.09845E-01 0.03837 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52108E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.68026E-01 0.00044  6.03159E-02 0.00043  1.97272E-04 0.00903 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.68583E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.68846E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.68583E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06521E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05936E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06090E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28950E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28725E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49106E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49014E-01 0.00051 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.37220E-03 0.00522  2.69564E-04 0.02070  8.34589E-04 0.01019  6.24093E-04 0.01280  1.31493E-03 0.00854  2.77200E-04 0.01961  5.18195E-05 0.04610 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42518E-01 0.01440  1.24441E-02 0.00251  3.23327E-02 9.5E-05  1.06051E-01 0.00058  2.97289E-01 0.00025  1.22610E+00 0.00436  4.51619E+00 0.04215 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25635E-03 0.00773  2.64396E-04 0.03065  8.04642E-04 0.01662  6.04904E-04 0.01857  1.27211E-03 0.01246  2.59957E-04 0.02763  5.03415E-05 0.06500 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.36348E-01 0.01998  1.24757E-02 4.0E-05  3.23309E-02 0.00014  1.06045E-01 0.00083  2.97204E-01 0.00035  1.23538E+00 0.00034  6.36165E+00 0.02864 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68268E-07 0.00126  8.67998E-07 0.00126  9.42718E-07 0.01596 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.40446E-07 0.00119  8.40185E-07 0.00119  9.12559E-07 0.01597 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26392E-03 0.00921  2.71331E-04 0.03238  8.13541E-04 0.01822  6.09142E-04 0.02089  1.27195E-03 0.01482  2.47857E-04 0.03272  5.01034E-05 0.07347 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.37896E-01 0.02524  1.24759E-02 4.7E-05  3.23285E-02 0.00014  1.05986E-01 0.00096  2.97246E-01 0.00043  1.23465E+00 0.00045  6.48513E+00 0.04060 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.71042E-07 0.01718  7.70890E-07 0.01718  7.50562E-07 0.05296 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.46494E-07 0.01718  7.46347E-07 0.01718  7.26537E-07 0.05292 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87431E-03 0.04155  2.52604E-04 0.13221  6.74092E-04 0.08227  5.33324E-04 0.09496  1.13849E-03 0.06217  2.42777E-04 0.14011  3.30272E-05 0.36232 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.17877E-01 0.09453  1.24771E-02 0.00011  3.23263E-02 0.00054  1.05919E-01 0.00291  2.96565E-01 0.00136  1.23387E+00 0.00120  5.27092E+00 0.16481 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85753E-03 0.04098  2.44884E-04 0.12866  6.63391E-04 0.08361  5.31297E-04 0.09258  1.15106E-03 0.06073  2.34232E-04 0.13603  3.26657E-05 0.34686 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.17657E-01 0.09078  1.24770E-02 0.00011  3.23257E-02 0.00053  1.05899E-01 0.00288  2.96584E-01 0.00136  1.23387E+00 0.00120  5.27092E+00 0.16481 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.76340E+03 0.03868 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.63504E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.35828E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29078E-03 0.00535 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81187E+03 0.00540 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94447E-09 0.00085 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84848E-05 0.00593  6.85071E-05 0.00596  9.98510E-07 0.41203 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.94379E-05 0.02288  5.92576E-05 0.02279  1.67993E-06 0.57000 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.21215E-04 0.02030  2.21365E-04 0.02039  1.80648E-04 0.40657 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70720E+01 0.01134 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49030E+01 0.00027  4.07556E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 17:37:32 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01085E+00  1.00492E+00  9.98279E-01  1.00392E+00  9.95673E-01  9.97587E-01  9.89442E-01  1.00322E+00  9.95268E-01  9.96104E-01  9.96627E-01  9.97156E-01  9.89135E-01  1.00676E+00  1.00223E+00  1.00697E+00  9.97554E-01  1.00830E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.28019E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67198E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39561E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53291E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61940E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51241E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50333E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24354E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66996E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499999 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50026E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50026E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.56381E+03 ;
RUNNING_TIME              (idx, 1)        =  1.58121E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.19067E-01  1.39533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.56251E+02  3.57205E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.27000E-02  1.80333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.03400E-01  5.09667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.58116E+02  5.04237E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88994 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98447E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90236E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.52466E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66410E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.52730E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91215E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33394E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58094E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49910E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.98262E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.67888E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.27732E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.25015E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69960E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85376E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93702E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32695E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31867E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52236E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34270E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81488E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07454E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51327E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50912E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70442E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.89777E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.28687E-03 -5.04934E+26  2.21302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.34144E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.59468E+18 0.00419  1.70005E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  9.16162E+19 0.00051  9.76796E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.64683E+17 0.01217  1.75570E-03 0.01215 ];
TH232_CAPT                (idx, [1:   4]) = [  9.70015E+19 0.00051  7.38021E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33723E+19 0.00138  1.01742E-01 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  5.76699E+16 0.02127  4.38748E-04 0.02126 ];
SM149_CAPT                (idx, [1:   4]) = [  1.38437E+17 0.01325  1.05332E-03 0.01326 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001028 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14139E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001028 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5305969 5.31173E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3786661 3.79048E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908398 9.09205E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001028 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.30968E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36402E+20 3.6E-06  2.36402E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.1E-07  9.37729E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31362E+20 0.00030  1.17374E+20 0.00022  1.39889E+19 0.00194 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.25135E+20 0.00018  2.11147E+20 0.00012  1.39889E+19 0.00194 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47444E+20 0.00036  2.47444E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29308E+22 0.00025  3.12411E+22 0.00023  1.68964E+21 0.00187 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24987E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47634E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11553E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51181E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51181E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.70497E-01 0.05084 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.29115E-01 0.01263 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.03083E-04 0.02430 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.01958E+03 0.02538 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09083E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.51734E-01 0.03920 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.92463E-01 0.03920 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52101E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.55796E-01 0.00043  5.95315E-02 0.00042  1.92403E-04 0.00913 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55762E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.55424E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55762E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05139E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06131E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05886E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28692E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28985E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51161E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51372E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.37967E-03 0.00573  2.71233E-04 0.01956  8.41178E-04 0.01109  6.10334E-04 0.01311  1.32615E-03 0.00932  2.78032E-04 0.02007  5.27382E-05 0.04583 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47829E-01 0.01570  1.24754E-02 3.4E-05  3.23313E-02 9.8E-05  1.06168E-01 0.00059  2.97494E-01 0.00027  1.23012E+00 0.00356  4.51661E+00 0.04211 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23109E-03 0.00811  2.61288E-04 0.02779  7.92836E-04 0.01501  5.75212E-04 0.01915  1.28832E-03 0.01323  2.64263E-04 0.02907  4.91739E-05 0.06372 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48107E-01 0.02228  1.24756E-02 4.4E-05  3.23283E-02 0.00014  1.06155E-01 0.00079  2.97447E-01 0.00037  1.23652E+00 0.00038  6.39670E+00 0.02784 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82489E-07 0.00124  8.82216E-07 0.00125  9.70819E-07 0.01887 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.43416E-07 0.00117  8.43154E-07 0.00117  9.28096E-07 0.01893 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22336E-03 0.00936  2.53283E-04 0.03215  8.02150E-04 0.01868  5.89844E-04 0.02132  1.26965E-03 0.01528  2.58105E-04 0.03173  5.03284E-05 0.07280 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.45829E-01 0.02545  1.24755E-02 5.7E-05  3.23218E-02 0.00015  1.06042E-01 0.00097  2.97648E-01 0.00042  1.23617E+00 0.00051  6.45431E+00 0.04046 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.79069E-07 0.01712  7.78980E-07 0.01712  7.71466E-07 0.05575 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.44580E-07 0.01711  7.44497E-07 0.01711  7.36468E-07 0.05558 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92441E-03 0.04224  2.30218E-04 0.13117  7.16574E-04 0.08495  4.59439E-04 0.09973  1.22824E-03 0.06318  2.41794E-04 0.14095  4.81359E-05 0.35759 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.06041E-01 0.08200  1.24759E-02 0.00013  3.23457E-02 0.00061  1.05806E-01 0.00291  2.97846E-01 0.00158  1.23622E+00 0.00159  5.76615E+00 0.15979 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92594E-03 0.04207  2.33515E-04 0.12627  7.06614E-04 0.08306  4.56079E-04 0.10036  1.23581E-03 0.06262  2.43402E-04 0.13923  5.05254E-05 0.35823 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.06210E-01 0.08274  1.24758E-02 0.00013  3.23487E-02 0.00060  1.05834E-01 0.00295  2.97835E-01 0.00156  1.23611E+00 0.00159  5.76615E+00 0.15979 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77032E+03 0.03921 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.76748E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.37923E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30000E-03 0.00554 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.76490E+03 0.00562 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92547E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75392E-05 0.00901  6.75088E-05 0.00900  1.36548E-06 0.38273 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.87692E-05 0.02814  5.87536E-05 0.02815  1.13834E-06 0.44389 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.95032E-04 0.02315  1.94996E-04 0.02317  2.05569E-04 0.37721 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72708E+01 0.01208 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50333E+01 0.00027  4.08710E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 18:13:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01843E+00  1.00202E+00  9.92282E-01  1.00869E+00  9.99632E-01  9.97868E-01  9.90009E-01  1.00624E+00  9.89695E-01  1.00451E+00  9.93282E-01  9.98051E-01  9.90675E-01  1.00368E+00  9.92935E-01  1.00228E+00  9.96914E-01  1.01282E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.27570E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67243E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39446E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53162E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61743E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52466E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51556E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27059E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.67199E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500684 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50038E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50038E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.92488E+03 ;
RUNNING_TIME              (idx, 1)        =  1.94406E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.58800E-01  1.39733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.92363E+02  3.61114E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.05167E-02  1.78167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.52767E-01  4.93167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.94401E+02  5.17304E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90137 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98451E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91426E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.59378E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67389E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.30439E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94788E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35757E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61388E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50609E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09586E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.78974E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.68810E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.41397E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25569E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94805E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35248E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33745E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32641E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53285E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.01358E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17352E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11472E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54630E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.33923E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.80736E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.99588E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.42937E-03 -7.57196E+26  2.21554E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.52304E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.61095E+18 0.00372  1.71754E-02 0.00371 ];
U233_FISS                 (idx, [1:   4]) = [  9.12421E+19 0.00051  9.72767E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.41836E+17 0.00846  3.64459E-03 0.00846 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81057E+19 0.00055  7.34158E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33030E+19 0.00138  9.95536E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  1.22630E+17 0.01389  9.18079E-04 0.01397 ];
SM149_CAPT                (idx, [1:   4]) = [  1.57257E+17 0.01296  1.17661E-03 0.01292 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001513 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15080E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001513 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5341983 5.34743E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3749798 3.75346E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909732 9.10619E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001513 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.43424E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36386E+20 3.5E-06  2.36386E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37700E+19 1.2E-07  9.37700E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33702E+20 0.00031  1.19522E+20 0.00021  1.41804E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27472E+20 0.00019  2.13292E+20 0.00012  1.41804E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49897E+20 0.00038  2.49897E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.33278E+22 0.00028  3.16146E+22 0.00024  1.71320E+21 0.00186 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.27570E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.50229E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12965E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50797E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50797E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.91606E-01 0.05233 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.39081E-01 0.01173 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93746E-04 0.02452 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.46110E+03 0.02302 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08942E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.16922E-01 0.04152 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.60695E-01 0.04152 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52092E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.46012E-01 0.00044  5.89455E-02 0.00043  1.93678E-04 0.00893 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.45794E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.45990E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.45794E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04053E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06154E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06006E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28670E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28833E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53627E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53424E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.45738E-03 0.00550  2.77784E-04 0.02067  8.39009E-04 0.01111  6.36690E-04 0.01240  1.36633E-03 0.00880  2.82195E-04 0.01935  5.53678E-05 0.04332 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45849E-01 0.01428  1.24133E-02 0.00355  3.23282E-02 9.4E-05  1.06117E-01 0.00059  2.97624E-01 0.00026  1.23664E+00 0.00029  4.58360E+00 0.04083 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22457E-03 0.00798  2.56428E-04 0.02930  7.94794E-04 0.01736  5.89142E-04 0.01917  1.27241E-03 0.01247  2.61210E-04 0.02814  5.05808E-05 0.06288 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46537E-01 0.02124  1.24759E-02 3.9E-05  3.23295E-02 0.00014  1.06143E-01 0.00084  2.97605E-01 0.00038  1.23704E+00 0.00035  6.40101E+00 0.02789 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.93766E-07 0.00127  8.93444E-07 0.00127  9.82423E-07 0.01626 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.45452E-07 0.00121  8.45148E-07 0.00121  9.29435E-07 0.01628 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26919E-03 0.00906  2.68428E-04 0.03149  8.10868E-04 0.01890  6.13383E-04 0.02084  1.25088E-03 0.01444  2.73947E-04 0.03189  5.16918E-05 0.07514 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53835E-01 0.02630  1.24754E-02 5.5E-05  3.23273E-02 0.00018  1.06125E-01 0.00096  2.97591E-01 0.00046  1.23735E+00 0.00054  6.86001E+00 0.03857 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.93194E-07 0.01721  7.93167E-07 0.01721  7.89966E-07 0.06051 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.50616E-07 0.01721  7.50590E-07 0.01721  7.47716E-07 0.06061 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.10818E-03 0.04086  2.59644E-04 0.12874  7.42896E-04 0.07704  6.07379E-04 0.07891  1.14982E-03 0.06403  2.92382E-04 0.13541  5.60602E-05 0.26834 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.82108E-01 0.09620  1.24769E-02 0.00012  3.23493E-02 0.00058  1.05531E-01 0.00223  2.97032E-01 0.00146  1.23857E+00 0.00176  6.11489E+00 0.11454 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09557E-03 0.03962  2.62373E-04 0.12668  7.47805E-04 0.07597  5.99181E-04 0.07928  1.13898E-03 0.06235  2.89359E-04 0.13239  5.78704E-05 0.27372 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.85531E-01 0.09665  1.24769E-02 0.00012  3.23505E-02 0.00059  1.05531E-01 0.00219  2.97076E-01 0.00146  1.23860E+00 0.00176  6.15594E+00 0.11401 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.94199E+03 0.03792 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.87981E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.39971E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28579E-03 0.00605 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.70128E+03 0.00610 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93481E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.64235E-05 0.01004  6.63201E-05 0.01033  7.45414E-07 0.46578 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.05513E-05 0.02613  6.04621E-05 0.02629  4.14440E-07 0.52043 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.86692E-04 0.02367  1.86835E-04 0.02376  1.48760E-04 0.44816 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71149E+01 0.01217 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51556E+01 0.00026  4.09062E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 18:50:57 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01109E+00  1.00894E+00  9.95080E-01  1.00222E+00  9.94943E-01  9.99470E-01  9.91401E-01  1.00199E+00  1.00027E+00  1.01198E+00  9.91499E-01  9.99274E-01  9.90741E-01  9.95504E-01  9.99255E-01  1.00926E+00  9.93335E-01  1.00375E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.27574E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67243E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38875E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52603E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61634E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53512E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52599E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.31003E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.67857E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500405 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50029E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50029E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.29437E+03 ;
RUNNING_TIME              (idx, 1)        =  2.31540E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.00187E+00  1.43067E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.29314E+02  3.69519E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.08550E-01  1.80333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.01200E-01  4.83833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.31534E+02  5.21714E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90920 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98469E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92257E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.62066E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67180E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.33521E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97428E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37324E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61398E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50208E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26773E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.92312E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.69214E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.56350E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93936E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06587E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07081E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30991E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30431E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50302E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.82825E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78705E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11905E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55434E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.93508E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85106E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.01440E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43284E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.80430E-03 -1.28157E+27  2.22079E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.81784E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.62155E+18 0.00384  1.72946E-02 0.00382 ];
U233_FISS                 (idx, [1:   4]) = [  9.03546E+19 0.00051  9.63646E-01 9.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.32356E+17 0.00566  8.87792E-03 0.00568 ];
PU239_FISS                (idx, [1:   4]) = [  2.59565E+13 1.00000  2.74907E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  9.98099E+19 0.00054  7.28816E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32162E+19 0.00138  9.65074E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  2.95166E+17 0.00946  2.15546E-03 0.00947 ];
PU239_CAPT                (idx, [1:   4]) = [  2.52006E+13 1.00000  1.85185E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.67691E+17 0.01226  1.22446E-03 0.01223 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001171 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13434E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001171 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5394479 5.40020E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3693698 3.69732E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912994 9.13819E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001171 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.48897E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36352E+20 3.7E-06  2.36352E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37626E+19 1.1E-07  9.37626E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37031E+20 0.00030  1.22628E+20 0.00022  1.44034E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30794E+20 0.00018  2.16390E+20 0.00012  1.44034E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53600E+20 0.00037  2.53600E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.39101E+22 0.00025  3.21798E+22 0.00023  1.73025E+21 0.00187 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.31757E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53969E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14901E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50043E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50043E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.84981E-01 0.05055 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31278E-01 0.01217 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.84145E-04 0.02313 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.25219E+03 0.02256 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08621E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.35939E-01 0.03920 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.77864E-01 0.03920 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52075E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.31987E-01 0.00042  5.80630E-02 0.00040  1.87984E-04 0.00911 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.31722E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.32037E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.31722E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02540E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05427E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05451E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29607E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29550E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.58041E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.57981E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.55459E-03 0.00579  2.84151E-04 0.01874  8.66068E-04 0.01122  6.58283E-04 0.01259  1.38927E-03 0.00918  2.97442E-04 0.01988  5.93798E-05 0.04146 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.58848E-01 0.01508  1.24449E-02 0.00251  3.23171E-02 0.00011  1.06170E-01 0.00053  2.98070E-01 0.00028  1.23381E+00 0.00357  4.98234E+00 0.03750 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25513E-03 0.00794  2.63282E-04 0.02854  7.85190E-04 0.01576  6.06667E-04 0.01944  1.26783E-03 0.01301  2.77653E-04 0.02706  5.45102E-05 0.06384 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.59776E-01 0.02164  1.24760E-02 3.7E-05  3.23221E-02 0.00015  1.06181E-01 0.00078  2.98098E-01 0.00039  1.23978E+00 0.00054  6.54854E+00 0.02624 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.09193E-07 0.00134  9.08911E-07 0.00134  9.99801E-07 0.02022 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.47297E-07 0.00127  8.47034E-07 0.00128  9.31634E-07 0.02022 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24424E-03 0.00898  2.63164E-04 0.03188  7.64327E-04 0.01949  6.13470E-04 0.01996  1.26762E-03 0.01507  2.80086E-04 0.03106  5.55795E-05 0.07072 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.56937E-01 0.02427  1.24760E-02 5.0E-05  3.23208E-02 0.00019  1.06268E-01 0.00100  2.98185E-01 0.00048  1.24006E+00 0.00079  6.22465E+00 0.03993 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.01471E-07 0.01710  8.01165E-07 0.01710  9.42686E-07 0.05563 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.47049E-07 0.01709  7.46764E-07 0.01710  8.78814E-07 0.05559 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83953E-03 0.04162  2.42847E-04 0.14339  7.31582E-04 0.07326  5.17916E-04 0.08953  1.09546E-03 0.06100  2.10109E-04 0.12858  4.16191E-05 0.31135 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.19077E-01 0.09610  1.24777E-02 9.5E-05  3.23251E-02 0.00056  1.06233E-01 0.00326  2.98099E-01 0.00161  1.23812E+00 0.00261  6.20925E+00 0.12994 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.86541E-03 0.04129  2.38967E-04 0.13953  7.39229E-04 0.07248  5.25493E-04 0.08760  1.11332E-03 0.06088  2.05740E-04 0.12867  4.26591E-05 0.30909 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.20324E-01 0.10024  1.24777E-02 9.5E-05  3.23221E-02 0.00055  1.06243E-01 0.00324  2.98086E-01 0.00158  1.23814E+00 0.00261  6.20925E+00 0.12994 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.55325E+03 0.03757 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.01730E-07 0.00062 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.40348E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.27362E-03 0.00627 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.63109E+03 0.00631 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95330E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73200E-05 0.00868  6.71360E-05 0.00906  1.04921E-06 0.40664 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.05534E-05 0.02464  6.00421E-05 0.02426  1.31217E-06 0.55795 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77507E-04 0.02294  1.77539E-04 0.02301  1.64674E-04 0.40847 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70485E+01 0.01068 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52599E+01 0.00026  4.08993E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 19:28:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01319E+00  9.95330E-01  9.86431E-01  1.00851E+00  9.95539E-01  1.00037E+00  9.90528E-01  1.00291E+00  9.97010E-01  9.99898E-01  9.99074E-01  1.00819E+00  9.92834E-01  1.00268E+00  1.00218E+00  1.00787E+00  9.89887E-01  1.00756E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.27959E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67204E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38591E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52340E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61728E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54082E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53171E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33009E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.68382E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500339 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50045E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50045E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.66496E+03 ;
RUNNING_TIME              (idx, 1)        =  2.69004E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.14860E+00  1.46733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.66592E+02  3.72771E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.26750E-01  1.82000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.79383E-01  2.78133E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.68772E+02  5.28589E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90676 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98312E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92006E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.70049E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68496E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.05083E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00472E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38892E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.66269E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51305E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75362E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22443E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.14205E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.82294E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23130E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33892E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32959E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30587E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29785E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49598E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.54926E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27973E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.15703E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.65959E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.85665E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.93967E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03212E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.18245E-02 -2.61082E+27  2.23408E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01813E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.63688E+18 0.00369  1.74667E-02 0.00358 ];
U233_FISS                 (idx, [1:   4]) = [  8.80251E+19 0.00052  9.39389E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.33268E+18 0.00333  2.48945E-02 0.00331 ];
PU239_FISS                (idx, [1:   4]) = [  6.95603E+14 0.20682  7.41355E-06 0.20689 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00830E+20 0.00054  7.14778E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.29052E+19 0.00138  9.14866E-02 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  8.25803E+17 0.00547  5.85470E-03 0.00550 ];
PU239_CAPT                (idx, [1:   4]) = [  2.57101E+14 0.31268  1.82078E-06 0.31268 ];
PU240_CAPT                (idx, [1:   4]) = [  2.54514E+13 1.00000  1.81449E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74754E+17 0.01180  1.23906E-03 0.01181 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001806 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13677E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001806 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5461568 5.46703E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3628283 3.63158E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911955 9.12757E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001806 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.99886E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36253E+20 3.6E-06  2.36253E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37405E+19 1.1E-07  9.37405E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.41031E+20 0.00029  1.26354E+20 0.00020  1.46771E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34772E+20 0.00018  2.20095E+20 0.00012  1.46771E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58029E+20 0.00037  2.58029E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45537E+22 0.00026  3.27923E+22 0.00023  1.76142E+21 0.00195 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35527E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58324E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17057E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.48114E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.48114E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.22953E-01 0.05232 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19730E-01 0.01344 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90160E-04 0.02419 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.19650E+03 0.02294 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08728E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.99276E-01 0.04131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.44543E-01 0.04131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52029E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99748E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.15120E-01 0.00042  5.70139E-02 0.00041  1.90208E-04 0.00851 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.15613E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.15654E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.15613E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00758E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02574E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02565E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33358E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33344E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.67767E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.67569E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.67121E-03 0.00554  2.81152E-04 0.01972  8.95384E-04 0.01204  6.69351E-04 0.01247  1.45013E-03 0.00891  3.15367E-04 0.01835  5.98230E-05 0.04259 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.58675E-01 0.01469  1.24447E-02 0.00251  3.23010E-02 0.00011  1.06306E-01 0.00057  2.98883E-01 0.00029  1.24599E+00 0.00056  4.68321E+00 0.03908 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.36817E-03 0.00788  2.59157E-04 0.02951  8.28558E-04 0.01597  6.08060E-04 0.01836  1.33500E-03 0.01217  2.79492E-04 0.02752  5.79042E-05 0.06393 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.61117E-01 0.02272  1.24761E-02 3.6E-05  3.23022E-02 0.00015  1.06205E-01 0.00070  2.98851E-01 0.00042  1.24568E+00 0.00069  6.49328E+00 0.02560 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.18936E-07 0.00124  9.18447E-07 0.00124  1.05837E-06 0.02035 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.40882E-07 0.00118  8.40435E-07 0.00118  9.68381E-07 0.02035 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.32525E-03 0.00854  2.48946E-04 0.03318  8.00140E-04 0.01818  6.15591E-04 0.02271  1.32064E-03 0.01402  2.87390E-04 0.03094  5.25429E-05 0.07416 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.56377E-01 0.02508  1.24752E-02 6.2E-05  3.22918E-02 0.00019  1.06359E-01 0.00098  2.98720E-01 0.00050  1.24537E+00 0.00096  6.55064E+00 0.03838 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.12550E-07 0.01708  8.11781E-07 0.01708  9.46111E-07 0.05776 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.43584E-07 0.01707  7.42881E-07 0.01708  8.65632E-07 0.05768 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01783E-03 0.04367  1.86942E-04 0.14459  7.61349E-04 0.08307  4.74617E-04 0.09557  1.30650E-03 0.06138  2.28945E-04 0.14481  5.94828E-05 0.26846 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.19409E-01 0.11803  1.24761E-02 0.00016  3.22790E-02 0.00061  1.06703E-01 0.00368  2.98908E-01 0.00168  1.24816E+00 0.00306  6.51068E+00 0.10138 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.04459E-03 0.04318  1.91790E-04 0.14551  7.65806E-04 0.08250  4.62343E-04 0.09215  1.33147E-03 0.06062  2.33765E-04 0.14168  5.94136E-05 0.26877 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.06511E-01 0.11236  1.24761E-02 0.00016  3.22780E-02 0.00060  1.06719E-01 0.00369  2.98822E-01 0.00166  1.24825E+00 0.00307  6.51068E+00 0.10138 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75015E+03 0.04043 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.11173E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33773E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40157E-03 0.00574 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.73346E+03 0.00572 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.96460E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68688E-05 0.00973  6.66405E-05 0.01006  1.33520E-06 0.38782 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.73187E-05 0.02804  5.67103E-05 0.02802  2.63641E-06 0.41959 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.81981E-04 0.02411  1.81946E-04 0.02420  1.85750E-04 0.37525 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70339E+01 0.01175 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53171E+01 0.00027  4.06955E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 20:06:12 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01854E+00  9.99571E-01  1.00035E+00  1.01611E+00  9.97030E-01  1.00044E+00  9.93196E-01  1.00331E+00  9.95417E-01  9.94189E-01  9.91230E-01  9.99460E-01  9.93399E-01  1.01127E+00  9.99512E-01  9.97644E-01  9.85711E-01  1.00361E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.28987E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67101E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38653E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52434E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61707E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52458E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51549E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.29686E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.68457E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500789 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50083E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50083E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.03216E+03 ;
RUNNING_TIME              (idx, 1)        =  3.06796E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.29713E+00  1.48533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.04191E+02  3.75993E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.46017E-01  1.92667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.70400E+00  1.02457E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.05815E+02  5.30485E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88330 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98425E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89389E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.87424E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73076E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59515E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01886E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39294E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82152E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55774E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.13437E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77546E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.05329E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20639E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04599E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84790E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38249E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38028E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34889E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56917E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.83067E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30609E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26308E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91804E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.43151E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10178E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03884E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.23739E-02 -4.94009E+27  2.25737E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02853E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.59274E+18 0.00398  1.69978E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  8.46380E+19 0.00054  9.03271E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.73702E+18 0.00239  5.05550E-02 0.00235 ];
PU239_FISS                (idx, [1:   4]) = [  2.13574E+16 0.03619  2.27791E-04 0.03620 ];
PU240_FISS                (idx, [1:   4]) = [  5.43418E+14 0.21269  5.80735E-06 0.21269 ];
PU241_FISS                (idx, [1:   4]) = [  5.95508E+14 0.20270  6.33008E-06 0.20270 ];
TH232_CAPT                (idx, [1:   4]) = [  9.88072E+19 0.00051  6.92610E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23759E+19 0.00142  8.67514E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.69102E+18 0.00391  1.18533E-02 0.00387 ];
U238_CAPT                 (idx, [1:   4]) = [  1.03877E+14 0.49813  7.26801E-07 0.49813 ];
PU239_CAPT                (idx, [1:   4]) = [  9.66497E+15 0.05266  6.77266E-05 0.05259 ];
PU240_CAPT                (idx, [1:   4]) = [  1.48351E+15 0.12789  1.03833E-05 0.12789 ];
PU241_CAPT                (idx, [1:   4]) = [  1.55349E+14 0.40572  1.08569E-06 0.40568 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74452E+17 0.01190  1.22273E-03 0.01187 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003304 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11068E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003304 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5488570 5.49306E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605284 3.60795E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909450 9.10094E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003304 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.52737E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36126E+20 3.9E-06  2.36126E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37022E+19 1.1E-07  9.37022E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42612E+20 0.00028  1.27824E+20 0.00020  1.47885E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36314E+20 0.00017  2.21526E+20 0.00012  1.47885E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59711E+20 0.00034  2.59711E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46925E+22 0.00025  3.29152E+22 0.00022  1.77728E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36370E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59951E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17396E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42754E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42754E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.22041E-01 0.05457 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.10864E-01 0.01409 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.84219E-04 0.02444 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.37385E+03 0.02145 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08994E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.77772E-01 0.04282 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.25276E-01 0.04282 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51997E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99830E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09083E-01 0.00042  5.66353E-02 0.00042  1.90286E-04 0.00945 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09382E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09233E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09382E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00045E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.96940E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.96673E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.41085E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.41435E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.81188E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81192E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.80853E-03 0.00570  2.78994E-04 0.01985  9.11641E-04 0.01078  6.91687E-04 0.01289  1.50701E-03 0.00838  3.45828E-04 0.01873  7.33759E-05 0.04075 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.93461E-01 0.01519  1.24761E-02 3.1E-05  3.22665E-02 0.00011  1.06537E-01 0.00058  3.00025E-01 0.00030  1.25343E+00 0.00069  5.43202E+00 0.03322 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.40381E-03 0.00814  2.42668E-04 0.02912  8.19046E-04 0.01638  6.22252E-04 0.01818  1.34200E-03 0.01271  3.13324E-04 0.02624  6.45211E-05 0.06199 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.98404E-01 0.02246  1.24760E-02 4.1E-05  3.22699E-02 0.00017  1.06480E-01 0.00080  2.99881E-01 0.00042  1.25410E+00 0.00089  7.09729E+00 0.02186 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.98880E-07 0.00126  8.98511E-07 0.00127  1.00474E-06 0.01804 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17085E-07 0.00115  8.16749E-07 0.00116  9.13355E-07 0.01804 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.35069E-03 0.00962  2.40258E-04 0.03341  7.98595E-04 0.01924  6.08350E-04 0.02180  1.32066E-03 0.01528  3.13146E-04 0.03144  6.96766E-05 0.06447 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.12944E-01 0.02627  1.24766E-02 5.1E-05  3.22624E-02 0.00021  1.06478E-01 0.00112  2.99862E-01 0.00053  1.25317E+00 0.00126  7.13708E+00 0.02931 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.92398E-07 0.01703  7.91921E-07 0.01703  8.53209E-07 0.05322 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20562E-07 0.01702  7.20130E-07 0.01702  7.75731E-07 0.05321 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.07986E-03 0.04218  2.17041E-04 0.14883  6.53135E-04 0.08406  5.69232E-04 0.09712  1.29583E-03 0.06550  2.91765E-04 0.12844  5.28547E-05 0.23471 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.34363E-01 0.11212  1.24777E-02 0.00012  3.22305E-02 0.00065  1.06349E-01 0.00308  3.00317E-01 0.00189  1.24498E+00 0.00267  7.39771E+00 0.08833 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09265E-03 0.04148  2.21605E-04 0.14430  6.67986E-04 0.08270  5.80629E-04 0.09620  1.28101E-03 0.06490  2.89640E-04 0.12415  5.17760E-05 0.23915 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.27920E-01 0.11023  1.24777E-02 0.00012  3.22295E-02 0.00066  1.06359E-01 0.00308  3.00342E-01 0.00187  1.24498E+00 0.00267  7.39771E+00 0.08833 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.90256E+03 0.03861 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.91960E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10812E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43069E-03 0.00628 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84737E+03 0.00636 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92797E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68316E-05 0.00952  6.68196E-05 0.00956  1.63028E-06 0.36081 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.77252E-05 0.02835  5.76964E-05 0.02844  1.04672E-06 0.47909 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.78338E-04 0.02380  1.78219E-04 0.02391  2.28521E-04 0.35223 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65176E+01 0.01108 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51549E+01 0.00026  4.01919E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 20:44:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01451E+00  1.00945E+00  9.89693E-01  9.99172E-01  1.00138E+00  9.98525E-01  9.86885E-01  1.00576E+00  9.91862E-01  1.00506E+00  9.96546E-01  9.98544E-01  9.89301E-01  1.01075E+00  1.00165E+00  1.00657E+00  9.94063E-01  1.00029E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.29518E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67048E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38619E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52415E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61556E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51313E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50403E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27651E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.68510E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500346 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50062E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50062E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.40346E+03 ;
RUNNING_TIME              (idx, 1)        =  3.44783E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.44880E+00  1.51667E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.41978E+02  3.77869E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.66567E-01  2.05500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.48488E+00  7.80800E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.44048E+02  5.32906E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87131 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98409E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88049E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.90325E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73846E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.71285E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01307E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38837E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85610E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56570E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.08549E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98966E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.91052E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.39126E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13506E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04266E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81896E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38569E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35163E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57364E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88018E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73617E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27640E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97435E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.85824E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11912E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03883E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.74769E-02 -6.06682E+27  2.26864E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02680E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.58979E+18 0.00400  1.69643E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  8.35758E+19 0.00054  8.91856E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  5.41533E+18 0.00218  5.77888E-02 0.00213 ];
PU239_FISS                (idx, [1:   4]) = [  4.87410E+16 0.02296  5.19908E-04 0.02289 ];
PU240_FISS                (idx, [1:   4]) = [  1.76668E+15 0.12012  1.88410E-05 0.12009 ];
PU241_FISS                (idx, [1:   4]) = [  2.15657E+15 0.10502  2.30268E-05 0.10504 ];
TH232_CAPT                (idx, [1:   4]) = [  9.76083E+19 0.00057  6.84365E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21925E+19 0.00149  8.54861E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  1.93308E+18 0.00361  1.35534E-02 0.00357 ];
U238_CAPT                 (idx, [1:   4]) = [  2.05722E+14 0.35047  1.44979E-06 0.35047 ];
PU239_CAPT                (idx, [1:   4]) = [  2.45771E+16 0.03223  1.72283E-04 0.03221 ];
PU240_CAPT                (idx, [1:   4]) = [  5.21937E+15 0.07222  3.66253E-05 0.07231 ];
PU241_CAPT                (idx, [1:   4]) = [  3.87537E+14 0.25365  2.72137E-06 0.25364 ];
SM149_CAPT                (idx, [1:   4]) = [  1.76280E+17 0.01306  1.23617E-03 0.01307 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002476 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10455E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002476 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486830 5.49180E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605412 3.60833E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910234 9.10912E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002476 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.26432E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36119E+20 4.2E-06  2.36119E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36879E+19 1.1E-07  9.36879E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42606E+20 0.00028  1.27860E+20 0.00021  1.47460E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36294E+20 0.00017  2.21548E+20 0.00012  1.47460E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59708E+20 0.00036  2.59708E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46380E+22 0.00025  3.28653E+22 0.00023  1.77274E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36582E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59952E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17095E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39932E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39932E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.57013E-01 0.05099 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.43952E-01 0.01314 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.78915E-04 0.02372 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.87869E+03 0.02375 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08912E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.98127E-01 0.04109 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.43548E-01 0.04109 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52027E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99861E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09398E-01 0.00045  5.66484E-02 0.00044  1.90429E-04 0.00910 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09346E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09219E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09346E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00050E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94452E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94335E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44647E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44780E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86889E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86293E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.79911E-03 0.00577  2.86917E-04 0.01942  8.99747E-04 0.01113  6.78658E-04 0.01297  1.51544E-03 0.00880  3.45935E-04 0.01807  7.24099E-05 0.03791 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.91568E-01 0.01429  1.24765E-02 6.0E-05  3.22548E-02 0.00013  1.06574E-01 0.00058  3.00310E-01 0.00031  1.25426E+00 0.00262  5.60779E+00 0.03129 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.37930E-03 0.00812  2.58301E-04 0.02835  7.95783E-04 0.01459  5.93897E-04 0.01866  1.36684E-03 0.01262  3.01177E-04 0.02701  6.33052E-05 0.05874 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.89273E-01 0.02213  1.24762E-02 4.0E-05  3.22523E-02 0.00017  1.06593E-01 0.00085  3.00295E-01 0.00045  1.25938E+00 0.00097  6.80800E+00 0.02216 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.88860E-07 0.00130  8.88530E-07 0.00131  9.89173E-07 0.01620 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.08265E-07 0.00123  8.07965E-07 0.00123  8.99580E-07 0.01622 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.33807E-03 0.00919  2.59676E-04 0.03482  7.80560E-04 0.01952  5.96301E-04 0.02219  1.34311E-03 0.01432  2.94157E-04 0.03023  6.42670E-05 0.06346 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.91860E-01 0.02533  1.24773E-02 0.00011  3.22507E-02 0.00024  1.06422E-01 0.00097  3.00217E-01 0.00055  1.25796E+00 0.00135  6.80888E+00 0.03220 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.80815E-07 0.01712  7.80532E-07 0.01713  8.15241E-07 0.05470 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.10190E-07 0.01711  7.09933E-07 0.01712  7.41617E-07 0.05474 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87140E-03 0.04518  2.38294E-04 0.12558  6.38011E-04 0.08491  4.94497E-04 0.09492  1.16608E-03 0.06655  2.74505E-04 0.13582  6.00098E-05 0.34413 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.23925E-01 0.12691  1.24764E-02 0.00014  3.22484E-02 0.00074  1.06936E-01 0.00349  3.01427E-01 0.00203  1.25525E+00 0.00355  6.59011E+00 0.10568 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.83493E-03 0.04450  2.33889E-04 0.12311  6.25398E-04 0.08368  4.95369E-04 0.09275  1.15410E-03 0.06546  2.69158E-04 0.13370  5.70100E-05 0.37380 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.19679E-01 0.12552  1.24764E-02 0.00014  3.22500E-02 0.00074  1.06938E-01 0.00349  3.01412E-01 0.00203  1.25534E+00 0.00355  6.58687E+00 0.10562 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.71469E+03 0.04296 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.81501E-07 0.00080 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.01561E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.33665E-03 0.00620 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.78603E+03 0.00624 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90436E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68744E-05 0.00793  6.68211E-05 0.00799  7.54506E-07 0.51222 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.74103E-05 0.02814  5.73549E-05 0.02820  5.93017E-07 0.59379 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.73082E-04 0.02299  1.73343E-04 0.02302  1.01573E-04 0.49951 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62883E+01 0.01146 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50403E+01 0.00027  3.99631E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 21:22:31 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00740E+00  1.01104E+00  9.93837E-01  1.00554E+00  9.97854E-01  9.99526E-01  9.93379E-01  1.00049E+00  9.94281E-01  1.00431E+00  9.93353E-01  9.99814E-01  9.92569E-01  1.00668E+00  9.95803E-01  9.98181E-01  9.97305E-01  1.00863E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.30097E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66990E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38706E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52519E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61752E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50656E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49746E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26066E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.68622E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500289 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50048E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50048E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.77792E+03 ;
RUNNING_TIME              (idx, 1)        =  3.83104E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.60222E+00  1.53417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.80096E+02  3.81179E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.85150E-01  1.85833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.28138E+00  7.96450E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.82353E+02  5.34188E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86135 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98475E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86935E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.90509E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73963E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.64797E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99885E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37844E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87207E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56777E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.17306E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17185E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.91796E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.56486E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21214E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20690E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20396E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37856E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34508E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56528E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90752E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11839E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27022E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99968E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.20112E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10125E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03938E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.26227E-02 -7.20299E+27  2.28000E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02420E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.58080E+18 0.00406  1.68586E-02 0.00397 ];
U233_FISS                 (idx, [1:   4]) = [  8.27865E+19 0.00056  8.82959E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.94267E+18 0.00211  6.33804E-02 0.00202 ];
PU239_FISS                (idx, [1:   4]) = [  8.47737E+16 0.01698  9.04214E-04 0.01698 ];
PU240_FISS                (idx, [1:   4]) = [  4.31406E+15 0.07895  4.60082E-05 0.07896 ];
PU241_FISS                (idx, [1:   4]) = [  3.76935E+15 0.08466  4.01252E-05 0.08470 ];
TH232_CAPT                (idx, [1:   4]) = [  9.66473E+19 0.00052  6.77215E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20820E+19 0.00149  8.46600E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.10483E+18 0.00347  1.47489E-02 0.00345 ];
U238_CAPT                 (idx, [1:   4]) = [  2.60124E+14 0.31267  1.81959E-06 0.31268 ];
PU239_CAPT                (idx, [1:   4]) = [  4.24864E+16 0.02425  2.97677E-04 0.02422 ];
PU240_CAPT                (idx, [1:   4]) = [  1.06589E+16 0.05193  7.47330E-05 0.05198 ];
PU241_CAPT                (idx, [1:   4]) = [  1.16792E+15 0.15725  8.18582E-06 0.15737 ];
SM149_CAPT                (idx, [1:   4]) = [  1.79833E+17 0.01178  1.26017E-03 0.01177 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001905 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09625E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001905 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487108 5.49226E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605139 3.60832E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909658 9.10377E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001905 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.73226E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36132E+20 4.3E-06  2.36132E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36753E+19 9.9E-08  9.36753E+19 9.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42716E+20 0.00029  1.27887E+20 0.00020  1.48296E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36392E+20 0.00018  2.21562E+20 0.00012  1.48296E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59846E+20 0.00034  2.59846E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46175E+22 0.00025  3.28377E+22 0.00022  1.77974E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36565E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60048E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16987E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.37174E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.37174E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.58976E-01 0.05099 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.20799E-01 0.01305 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.91671E-04 0.02307 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.12152E+03 0.02268 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08965E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.12821E-01 0.03982 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.57044E-01 0.03982 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52075E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99887E+02 9.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09474E-01 0.00045  5.66562E-02 0.00043  1.92606E-04 0.00859 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09051E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08779E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09051E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00013E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92430E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92474E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47604E-02 0.00149 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47500E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90899E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90284E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.87189E-03 0.00561  2.87787E-04 0.01921  9.22042E-04 0.01064  6.94916E-04 0.01301  1.54644E-03 0.00855  3.44743E-04 0.01708  7.59532E-05 0.03698 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.97416E-01 0.01488  1.24763E-02 3.0E-05  3.22453E-02 0.00013  1.06636E-01 0.00054  3.00590E-01 0.00031  1.25726E+00 0.00074  5.80505E+00 0.02979 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.41196E-03 0.00802  2.62078E-04 0.02791  8.15456E-04 0.01623  6.02333E-04 0.01971  1.36894E-03 0.01188  2.96517E-04 0.02485  6.66372E-05 0.05298 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.94886E-01 0.02167  1.24762E-02 3.9E-05  3.22546E-02 0.00019  1.06606E-01 0.00076  3.00509E-01 0.00043  1.25635E+00 0.00097  6.85431E+00 0.02184 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.78927E-07 0.00136  8.78604E-07 0.00136  9.72355E-07 0.01650 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.99286E-07 0.00126  7.98993E-07 0.00126  8.84255E-07 0.01650 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39005E-03 0.00867  2.53412E-04 0.03204  8.02007E-04 0.01798  6.03315E-04 0.02253  1.34374E-03 0.01423  3.22039E-04 0.02966  6.55321E-05 0.06491 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.05153E-01 0.02547  1.24770E-02 4.6E-05  3.22538E-02 0.00025  1.06623E-01 0.00104  3.00737E-01 0.00051  1.25574E+00 0.00136  7.11742E+00 0.02993 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.69466E-07 0.01714  7.69043E-07 0.01714  8.51049E-07 0.05290 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.99789E-07 0.01713  6.99406E-07 0.01713  7.73840E-07 0.05287 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.19986E-03 0.04055  2.42623E-04 0.14789  7.88675E-04 0.07590  4.88384E-04 0.09646  1.28927E-03 0.06439  3.06409E-04 0.12319  8.44941E-05 0.24472 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.92789E-01 0.11415  1.24763E-02 0.00014  3.22108E-02 0.00076  1.06478E-01 0.00297  3.00359E-01 0.00186  1.25787E+00 0.00335  7.52998E+00 0.07655 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.21800E-03 0.04002  2.47004E-04 0.14448  7.92344E-04 0.07551  4.86650E-04 0.09822  1.29693E-03 0.06300  3.13641E-04 0.12172  8.14322E-05 0.24933 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.84690E-01 0.11272  1.24763E-02 0.00014  3.22072E-02 0.00076  1.06479E-01 0.00297  3.00361E-01 0.00186  1.25840E+00 0.00337  7.52998E+00 0.07655 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.18741E+03 0.03735 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.72286E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.93254E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46109E-03 0.00592 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.96820E+03 0.00590 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90913E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71744E-05 0.00941  6.71477E-05 0.00942  1.77355E-06 0.32081 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.76770E-05 0.03256  5.76357E-05 0.03266  1.49794E-06 0.45498 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84109E-04 0.02273  1.83719E-04 0.02285  3.10562E-04 0.34316 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63622E+01 0.01036 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49746E+01 0.00027  3.97864E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 22:01:38 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00694E+00  1.00409E+00  9.96126E-01  9.99635E-01  1.00543E+00  1.00580E+00  9.82085E-01  1.00085E+00  9.88926E-01  1.00544E+00  9.98799E-01  1.00419E+00  9.87841E-01  1.01284E+00  1.00390E+00  1.00055E+00  9.97766E-01  9.98786E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.31930E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66807E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38183E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52065E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61899E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48788E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47876E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24273E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.69465E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500335 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50021E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50021E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.16007E+03 ;
RUNNING_TIME              (idx, 1)        =  4.22222E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.75675E+00  1.54533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.19007E+02  3.89114E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.04567E-01  1.94167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.10387E+00  8.22417E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.21442E+02  5.36318E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85282 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98513E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.85984E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.92743E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74578E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.97988E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96461E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35813E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92869E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57601E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.73596E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75305E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.24471E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03893E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44284E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70457E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34389E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37344E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33838E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55774E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95286E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27229E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26698E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05618E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41862E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07935E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04233E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.30752E-02 -1.17188E+28  2.32516E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02226E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.54773E+18 0.00421  1.65168E-02 0.00416 ];
U233_FISS                 (idx, [1:   4]) = [  8.07560E+19 0.00055  8.61829E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.90111E+18 0.00196  7.36483E-02 0.00188 ];
U238_FISS                 (idx, [1:   4]) = [  7.77100E+13 0.57593  8.30159E-07 0.57596 ];
PU239_FISS                (idx, [1:   4]) = [  3.03207E+17 0.00924  3.23521E-03 0.00917 ];
PU240_FISS                (idx, [1:   4]) = [  2.17571E+16 0.03461  2.32263E-04 0.03464 ];
PU241_FISS                (idx, [1:   4]) = [  3.41121E+16 0.02679  3.64045E-04 0.02679 ];
TH232_CAPT                (idx, [1:   4]) = [  9.45724E+19 0.00053  6.59629E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17995E+19 0.00141  8.23015E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  2.43488E+18 0.00312  1.69841E-02 0.00315 ];
U238_CAPT                 (idx, [1:   4]) = [  9.14732E+14 0.17174  6.37204E-06 0.17158 ];
PU239_CAPT                (idx, [1:   4]) = [  1.40581E+17 0.01373  9.80580E-04 0.01373 ];
PU240_CAPT                (idx, [1:   4]) = [  5.93413E+16 0.02098  4.13932E-04 0.02099 ];
PU241_CAPT                (idx, [1:   4]) = [  7.96888E+15 0.05486  5.55940E-05 0.05480 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80027E+17 0.01210  1.25556E-03 0.01207 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000827 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09644E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000827 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496362 5.50204E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3592380 3.59595E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912085 9.12970E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000827 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 0.00000E+00 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36286E+20 4.7E-06  2.36286E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36358E+19 9.7E-08  9.36358E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43227E+20 0.00027  1.28401E+20 0.00019  1.48259E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36862E+20 0.00016  2.22037E+20 0.00011  1.48259E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60582E+20 0.00035  2.60582E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46910E+22 0.00024  3.29127E+22 0.00022  1.77826E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37911E+19 0.00134 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60654E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16829E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26637E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26637E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.18359E-01 0.05241 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.37656E-01 0.01178 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.05434E-04 0.02476 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.90830E+03 0.02364 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08706E-01 0.00012 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.01563E-01 0.04067 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.46565E-01 0.04067 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52346E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99972E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07346E-01 0.00042  5.65193E-02 0.00042  1.94259E-04 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07512E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06807E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07512E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98770E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88120E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88078E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.54098E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.54128E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.00067E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.99999E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.91983E-03 0.00534  2.80332E-04 0.02027  9.09975E-04 0.01081  7.05169E-04 0.01282  1.59308E-03 0.00844  3.51453E-04 0.01775  7.98245E-05 0.03573 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.02161E-01 0.01435  1.24767E-02 6.7E-05  3.22308E-02 0.00014  1.06615E-01 0.00050  3.01215E-01 0.00032  1.26029E+00 0.00089  5.81597E+00 0.02849 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47967E-03 0.00753  2.39446E-04 0.03114  8.15670E-04 0.01525  6.05982E-04 0.01812  1.42819E-03 0.01213  3.18474E-04 0.02582  7.19080E-05 0.05447 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.10002E-01 0.02105  1.24768E-02 7.9E-05  3.22269E-02 0.00020  1.06601E-01 0.00073  3.01139E-01 0.00042  1.26024E+00 0.00120  6.80982E+00 0.02078 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64964E-07 0.00126  8.64548E-07 0.00125  9.88559E-07 0.01779 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.84779E-07 0.00122  7.84401E-07 0.00121  8.96822E-07 0.01779 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42728E-03 0.00899  2.35806E-04 0.03404  7.90555E-04 0.01870  6.20841E-04 0.02086  1.40211E-03 0.01422  3.08389E-04 0.02974  6.95742E-05 0.06036 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08335E-01 0.02371  1.24763E-02 0.00010  3.22329E-02 0.00027  1.06521E-01 0.00097  3.01250E-01 0.00052  1.26154E+00 0.00161  7.02586E+00 0.02925 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.62562E-07 0.01715  7.62267E-07 0.01714  8.04427E-07 0.06929 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.91970E-07 0.01713  6.91703E-07 0.01713  7.30304E-07 0.06961 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.18592E-03 0.04291  1.95494E-04 0.14281  6.40586E-04 0.08895  6.14524E-04 0.08856  1.36592E-03 0.05857  2.64566E-04 0.12203  1.04826E-04 0.23767 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.52649E-01 0.10474  1.24757E-02 0.00016  3.22361E-02 0.00081  1.06854E-01 0.00324  3.01133E-01 0.00186  1.26290E+00 0.00393  6.46607E+00 0.08690 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.18299E-03 0.04241  1.97471E-04 0.13828  6.38074E-04 0.08821  6.08630E-04 0.08539  1.37135E-03 0.05832  2.64083E-04 0.12078  1.03374E-04 0.23305 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.53600E-01 0.10231  1.24757E-02 0.00016  3.22346E-02 0.00080  1.06856E-01 0.00325  3.01155E-01 0.00185  1.26248E+00 0.00388  6.46607E+00 0.08690 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.21882E+03 0.03972 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.55631E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.76298E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44055E-03 0.00500 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.02233E+03 0.00511 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91745E-09 0.00087 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74139E-05 0.00829  6.74323E-05 0.00829  1.12448E-06 0.39134 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.68268E-05 0.02527  5.68842E-05 0.02532  1.07745E-06 0.51963 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.98583E-04 0.02384  1.98662E-04 0.02387  1.88361E-04 0.37668 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58285E+01 0.01094 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47876E+01 0.00026  3.93525E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 22:40:48 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01407E+00  1.00151E+00  9.90274E-01  1.00756E+00  1.00210E+00  1.00928E+00  9.88954E-01  1.00046E+00  9.93415E-01  9.99836E-01  9.96949E-01  1.00211E+00  9.92801E-01  9.94859E-01  1.00329E+00  1.00630E+00  9.90300E-01  1.00592E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.36593E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66341E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36841E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50892E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63213E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48583E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47666E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.28136E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.73399E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500264 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50043E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50043E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.54259E+03 ;
RUNNING_TIME              (idx, 1)        =  4.61401E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.91403E+00  1.57283E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.57974E+02  3.89670E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.24250E-01  1.96833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.94960E+00  8.45683E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.60599E+02  5.38776E+02 ];
CPU_USAGE                 (idx, 1)        = 9.84521 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98503E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.85142E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.91491E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74028E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.61104E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92343E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33745E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95736E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57260E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53795E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39117E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36328E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14172E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69275E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23881E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58125E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34644E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31671E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52871E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.98588E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59850E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23500E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03488E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.14749E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01815E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04523E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06498E-01 -2.35144E+28  2.44312E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01933E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.51042E+18 0.00446  1.61357E-02 0.00443 ];
U233_FISS                 (idx, [1:   4]) = [  7.93822E+19 0.00057  8.48047E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.13241E+18 0.00202  7.61942E-02 0.00192 ];
U238_FISS                 (idx, [1:   4]) = [  1.30246E+14 0.44500  1.39111E-06 0.44500 ];
PU239_FISS                (idx, [1:   4]) = [  6.16091E+17 0.00688  6.58110E-03 0.00682 ];
PU240_FISS                (idx, [1:   4]) = [  7.26282E+16 0.01978  7.76187E-04 0.01981 ];
PU241_FISS                (idx, [1:   4]) = [  1.24805E+17 0.01381  1.33317E-03 0.01378 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27641E+19 0.00055  6.44290E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15270E+19 0.00149  8.00620E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.52592E+18 0.00326  1.75430E-02 0.00320 ];
U238_CAPT                 (idx, [1:   4]) = [  2.74534E+15 0.10270  1.90613E-05 0.10262 ];
PU239_CAPT                (idx, [1:   4]) = [  2.92177E+17 0.00927  2.02957E-03 0.00930 ];
PU240_CAPT                (idx, [1:   4]) = [  1.91850E+17 0.01167  1.33239E-03 0.01164 ];
PU241_CAPT                (idx, [1:   4]) = [  2.82775E+16 0.03098  1.96325E-04 0.03094 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85763E+17 0.01186  1.28998E-03 0.01183 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001705 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07479E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001705 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5504794 5.51000E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3579175 3.58225E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 917736 9.18497E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001705 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.11062E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36589E+20 5.1E-06  2.36589E+20 5.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35953E+19 1.0E-07  9.35953E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43922E+20 0.00026  1.28647E+20 0.00018  1.52749E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37517E+20 0.00016  2.22242E+20 0.00011  1.52749E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61307E+20 0.00035  2.61307E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49857E+22 0.00024  3.31719E+22 0.00022  1.81379E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40023E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61519E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17097E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.05398E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.05398E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.54406E-01 0.05194 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13726E-01 0.01125 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.51827E-04 0.02125 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.95122E+03 0.02343 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08153E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.38538E-01 0.03755 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.79963E-01 0.03755 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52779E+00 5.1E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00058E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05484E-01 0.00044  5.64037E-02 0.00044  1.93780E-04 0.00869 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05672E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05452E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05672E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97300E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85224E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85291E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58620E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58486E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07207E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.06750E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96020E-03 0.00536  2.66462E-04 0.02131  9.31002E-04 0.01088  7.08145E-04 0.01205  1.59419E-03 0.00879  3.70583E-04 0.01803  8.98222E-05 0.03311 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.28348E-01 0.01404  1.24815E-02 0.00015  3.22060E-02 0.00015  1.06758E-01 0.00054  3.01321E-01 0.00030  1.25983E+00 0.00098  6.20228E+00 0.02589 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.50788E-03 0.00749  2.38193E-04 0.03164  8.28943E-04 0.01583  6.27914E-04 0.01714  1.41206E-03 0.01344  3.21776E-04 0.02511  7.89884E-05 0.04854 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.25652E-01 0.01933  1.24797E-02 0.00011  3.21932E-02 0.00022  1.06764E-01 0.00071  3.01272E-01 0.00044  1.25984E+00 0.00126  7.14461E+00 0.01852 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.62141E-07 0.00134  8.61759E-07 0.00134  9.74243E-07 0.01714 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.80585E-07 0.00124  7.80239E-07 0.00124  8.82232E-07 0.01718 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43156E-03 0.00860  2.35921E-04 0.03668  7.97284E-04 0.01830  6.22402E-04 0.02020  1.37763E-03 0.01394  3.18002E-04 0.02984  8.03177E-05 0.05768 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.34146E-01 0.02414  1.24813E-02 0.00023  3.22054E-02 0.00029  1.06799E-01 0.00099  3.01450E-01 0.00053  1.25843E+00 0.00194  7.20970E+00 0.02618 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.60283E-07 0.01718  7.59682E-07 0.01718  8.53802E-07 0.04990 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.88484E-07 0.01717  6.87942E-07 0.01718  7.73146E-07 0.04979 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.05179E-03 0.04236  2.47592E-04 0.13680  6.90503E-04 0.07947  5.64860E-04 0.09259  1.26916E-03 0.06013  2.06991E-04 0.13011  7.26833E-05 0.26823 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.32228E-01 0.12532  1.24800E-02 1.9E-05  3.21747E-02 0.00083  1.06916E-01 0.00336  3.01226E-01 0.00195  1.27330E+00 0.00452  7.92388E+00 0.07695 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03675E-03 0.04176  2.40488E-04 0.13433  6.98902E-04 0.07882  5.48557E-04 0.09134  1.27040E-03 0.05943  2.00247E-04 0.12655  7.81496E-05 0.27461 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.39124E-01 0.12681  1.24800E-02 1.9E-05  3.21753E-02 0.00083  1.06878E-01 0.00334  3.01281E-01 0.00194  1.27331E+00 0.00451  7.91472E+00 0.07689 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.04546E+03 0.03893 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.57016E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.75954E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40126E-03 0.00540 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.97011E+03 0.00550 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.02677E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92063E-05 0.00642  6.91770E-05 0.00644  1.41278E-06 0.35826 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.20842E-05 0.02138  6.20810E-05 0.02148  1.54466E-06 0.44829 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.41279E-04 0.02047  2.41440E-04 0.02054  2.20536E-04 0.35255 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58752E+01 0.01091 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47666E+01 0.00027  3.90636E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 23:20:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01476E+00  9.97473E-01  9.88986E-01  1.00560E+00  1.00581E+00  1.00235E+00  9.91795E-01  1.00340E+00  9.92351E-01  1.00527E+00  1.00024E+00  1.00429E+00  9.89404E-01  1.00616E+00  1.00338E+00  1.00266E+00  9.88947E-01  9.97146E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.41531E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65847E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35395E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49628E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64773E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49001E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48069E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33498E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.77882E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500599 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50060E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50060E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.92648E+03 ;
RUNNING_TIME              (idx, 1)        =  5.00689E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.07558E+00  1.61550E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.97044E+02  3.90696E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.43300E-01  1.90500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.75767E+00  8.08000E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.99925E+02  5.39073E+02 ];
CPU_USAGE                 (idx, 1)        = 9.83941 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98512E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84504E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.91178E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73845E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.41762E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91219E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33186E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96537E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57124E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64241E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56711E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.46013E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16064E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76484E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39506E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93171E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33905E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31120E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52107E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00177E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00879E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22484E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02350E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.35462E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99977E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04692E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61101E-01 -3.55706E+28  2.56368E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01319E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.51787E+18 0.00427  1.62163E-02 0.00425 ];
U233_FISS                 (idx, [1:   4]) = [  7.92515E+19 0.00056  8.46670E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.10736E+18 0.00182  7.59317E-02 0.00178 ];
U238_FISS                 (idx, [1:   4]) = [  1.31077E+14 0.44501  1.40993E-06 0.44500 ];
PU239_FISS                (idx, [1:   4]) = [  6.62589E+17 0.00634  7.07842E-03 0.00630 ];
PU240_FISS                (idx, [1:   4]) = [  8.37077E+16 0.01809  8.93747E-04 0.01798 ];
PU241_FISS                (idx, [1:   4]) = [  1.55047E+17 0.01264  1.65597E-03 0.01258 ];
TH232_CAPT                (idx, [1:   4]) = [  9.20758E+19 0.00055  6.39430E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15794E+19 0.00148  8.04155E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51236E+18 0.00329  1.74487E-02 0.00332 ];
U238_CAPT                 (idx, [1:   4]) = [  2.98519E+15 0.09113  2.07368E-05 0.09109 ];
PU239_CAPT                (idx, [1:   4]) = [  3.13691E+17 0.00947  2.17814E-03 0.00941 ];
PU240_CAPT                (idx, [1:   4]) = [  2.28074E+17 0.01104  1.58398E-03 0.01104 ];
PU241_CAPT                (idx, [1:   4]) = [  3.42421E+16 0.02909  2.37787E-04 0.02910 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80159E+17 0.01210  1.25121E-03 0.01211 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002380 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07124E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002380 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496814 5.50174E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3573630 3.57638E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931936 9.32591E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002380 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.60187E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.0E-09  3.52484E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36656E+20 5.1E-06  2.36656E+20 5.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35888E+19 1.1E-07  9.35888E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43991E+20 0.00027  1.28303E+20 0.00019  1.56887E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37580E+20 0.00016  2.21891E+20 0.00011  1.56887E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61731E+20 0.00035  2.61731E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.52944E+22 0.00025  3.34510E+22 0.00022  1.84337E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.44101E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61990E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17391E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87342E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87342E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.97689E-01 0.04269 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23544E-01 0.00922 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.11046E-04 0.01903 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.18327E+03 0.02552 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06746E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.38063E-01 0.02968 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.69321E-01 0.02968 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52868E+00 5.1E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04391E-01 0.00046  5.63265E-02 0.00045  1.94929E-04 0.00909 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.04291E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.04240E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.04291E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97309E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85056E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85090E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58901E-02 0.00151 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58804E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07828E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07728E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93721E-03 0.00562  2.88935E-04 0.01963  9.10720E-04 0.01116  7.07324E-04 0.01220  1.58522E-03 0.00827  3.64736E-04 0.01829  8.02713E-05 0.03598 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13832E-01 0.01424  1.24775E-02 5.9E-05  3.21940E-02 0.00016  1.06820E-01 0.00056  3.01274E-01 0.00030  1.25691E+00 0.00115  6.24506E+00 0.02711 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47845E-03 0.00767  2.49054E-04 0.02677  8.10657E-04 0.01617  6.08387E-04 0.01913  1.40709E-03 0.01153  3.25244E-04 0.02629  7.80106E-05 0.05198 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.26666E-01 0.02089  1.24779E-02 9.8E-05  3.21851E-02 0.00023  1.06724E-01 0.00072  3.01331E-01 0.00045  1.25482E+00 0.00148  7.19343E+00 0.01923 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.69104E-07 0.00140  8.68699E-07 0.00140  9.80460E-07 0.01710 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.85948E-07 0.00134  7.85581E-07 0.00134  8.86727E-07 0.01713 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46551E-03 0.00933  2.58515E-04 0.03409  8.08335E-04 0.01865  6.14388E-04 0.01957  1.40722E-03 0.01472  3.06777E-04 0.03097  7.02788E-05 0.06273 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06580E-01 0.02409  1.24769E-02 6.3E-05  3.21862E-02 0.00028  1.06668E-01 0.00100  3.01271E-01 0.00053  1.25859E+00 0.00180  7.24664E+00 0.02823 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63718E-07 0.01722  7.63501E-07 0.01723  7.95737E-07 0.04997 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.90790E-07 0.01722  6.90594E-07 0.01722  7.19923E-07 0.05001 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.02002E-03 0.04235  2.61618E-04 0.12216  6.91067E-04 0.08750  4.75436E-04 0.09267  1.29325E-03 0.06173  2.52611E-04 0.14266  4.60382E-05 0.26084 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.81739E-01 0.08072  1.24729E-02 0.00020  3.22149E-02 0.00098  1.06600E-01 0.00306  3.00845E-01 0.00187  1.25103E+00 0.00714  7.96620E+00 0.07902 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.01944E-03 0.04199  2.60647E-04 0.12028  6.86920E-04 0.08722  4.83409E-04 0.09058  1.28611E-03 0.06030  2.57206E-04 0.13661  4.51400E-05 0.26524 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.88936E-01 0.08116  1.24729E-02 0.00020  3.22109E-02 0.00098  1.06578E-01 0.00301  3.00781E-01 0.00185  1.25099E+00 0.00715  8.05240E+00 0.07503 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.98437E+03 0.03966 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60607E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78260E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.48649E-03 0.00597 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.05080E+03 0.00588 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16821E-09 0.00095 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.87648E-05 0.00588  6.87687E-05 0.00587  8.44902E-07 0.45809 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.08916E-05 0.01944  6.08621E-05 0.01943  8.89542E-07 0.60905 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.01052E-04 0.01839  3.01749E-04 0.01845  1.30473E-04 0.44928 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59897E+01 0.01111 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48069E+01 0.00028  3.90378E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:59:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 23:42:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995964892 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02535E+00  9.94839E-01  9.90272E-01  1.00674E+00  9.98401E-01  1.00256E+00  9.84489E-01  1.00937E+00  9.91487E-01  1.00069E+00  9.94460E-01  9.98191E-01  9.91063E-01  9.97878E-01  9.97166E-01  1.00139E+00  9.98329E-01  1.01731E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.47289E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65271E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.33929E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.48376E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65965E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49663E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48723E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.39327E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.82913E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500111 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50044E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50044E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.13781E+03 ;
RUNNING_TIME              (idx, 1)        =  5.22710E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03638E+00  1.03638E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.23638E+00  1.60800E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.18843E+02  2.17990E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.62717E-01  1.94167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.28333E-02  1.28333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.55913E+00  8.01417E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.21953E+02  5.21953E+02 ];
CPU_USAGE                 (idx, 1)        = 9.82916 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97585E+00 3.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.83437E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22596.53;
MEMSIZE                   (idx, 1)        = 22412.04;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 260.33;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

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

TOT_ACTIVITY              (idx, 1)        =  6.90793E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73770E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.64077E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90570E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32801E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96802E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57087E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65585E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61414E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47133E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16154E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78576E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44092E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03138E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33696E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30970E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51894E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00377E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13469E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21925E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02028E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.39177E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98726E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04849E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16306E-01 -4.77597E+28  2.68557E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01260E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.51223E+18 0.00420  1.61653E-02 0.00413 ];
U233_FISS                 (idx, [1:   4]) = [  7.91558E+19 0.00052  8.46216E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.10776E+18 0.00188  7.59838E-02 0.00179 ];
U238_FISS                 (idx, [1:   4]) = [  7.85520E+13 0.57591  8.41097E-07 0.57590 ];
PU239_FISS                (idx, [1:   4]) = [  6.63449E+17 0.00633  7.09298E-03 0.00634 ];
PU240_FISS                (idx, [1:   4]) = [  8.60744E+16 0.01736  9.20031E-04 0.01732 ];
PU241_FISS                (idx, [1:   4]) = [  1.59442E+17 0.01225  1.70503E-03 0.01232 ];
TH232_CAPT                (idx, [1:   4]) = [  9.18385E+19 0.00057  6.36774E-01 0.00037 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15488E+19 0.00148  8.00761E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48602E+18 0.00318  1.72376E-02 0.00317 ];
U238_CAPT                 (idx, [1:   4]) = [  2.93725E+15 0.09147  2.03400E-05 0.09152 ];
PU239_CAPT                (idx, [1:   4]) = [  3.14405E+17 0.00853  2.17989E-03 0.00852 ];
PU240_CAPT                (idx, [1:   4]) = [  2.34924E+17 0.01121  1.62887E-03 0.01120 ];
PU241_CAPT                (idx, [1:   4]) = [  3.39930E+16 0.02573  2.35728E-04 0.02576 ];
SM149_CAPT                (idx, [1:   4]) = [  1.87769E+17 0.01113  1.30186E-03 0.01110 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001776 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08117E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001776 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5497039 5.50220E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3565519 3.56862E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 939218 9.39991E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001776 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.95578E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.1E-09  3.52484E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36667E+20 4.9E-06  2.36667E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35881E+19 1.0E-07  9.35881E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44175E+20 0.00030  1.28086E+20 0.00020  1.60888E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37763E+20 0.00018  2.21674E+20 0.00011  1.60888E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62123E+20 0.00035  2.62123E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.56163E+22 0.00026  3.37513E+22 0.00023  1.86491E+21 0.00191 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46402E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62403E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17742E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.71564E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.71564E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.35760E-01 0.03635 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.14403E-01 0.00883 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.71003E-04 0.01730 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.66622E+03 0.02726 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06009E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.24276E-01 0.02286 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.46774E-01 0.02286 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52881E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00074E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.02337E-01 0.00041  5.62086E-02 0.00040  1.92124E-04 0.00876 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.02908E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.02927E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.02908E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96588E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85344E+00 0.00023 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85150E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58448E-02 0.00155 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58707E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07538E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07999E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96401E-03 0.00492  2.85694E-04 0.01941  9.21205E-04 0.01092  7.12616E-04 0.01249  1.58851E-03 0.00807  3.74451E-04 0.01735  8.15378E-05 0.03888 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08260E-01 0.01459  1.24500E-02 0.00251  3.21981E-02 0.00015  1.06853E-01 0.00057  3.01442E-01 0.00033  1.25716E+00 0.00109  5.79179E+00 0.02915 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47727E-03 0.00753  2.54162E-04 0.03079  7.96063E-04 0.01625  6.25384E-04 0.01768  1.39274E-03 0.01223  3.35533E-04 0.02444  7.33855E-05 0.05319 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.13309E-01 0.02034  1.24818E-02 0.00021  3.22009E-02 0.00021  1.06756E-01 0.00076  3.01406E-01 0.00046  1.25561E+00 0.00166  6.88008E+00 0.02067 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80474E-07 0.00144  8.80056E-07 0.00145  1.00407E-06 0.01748 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.94448E-07 0.00142  7.94071E-07 0.00143  9.05831E-07 0.01744 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40849E-03 0.00889  2.33679E-04 0.03493  7.91784E-04 0.01890  6.07501E-04 0.01983  1.37688E-03 0.01412  3.23793E-04 0.02913  7.48612E-05 0.06326 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08963E-01 0.02413  1.24776E-02 9.5E-05  3.21931E-02 0.00028  1.06765E-01 0.00100  3.01357E-01 0.00055  1.25630E+00 0.00233  6.62424E+00 0.03091 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.73845E-07 0.01732  7.73195E-07 0.01732  9.32385E-07 0.07208 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.98527E-07 0.01732  6.97941E-07 0.01732  8.40912E-07 0.07188 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.95719E-03 0.04126  2.12446E-04 0.16016  6.40290E-04 0.07540  5.16876E-04 0.09105  1.22324E-03 0.06454  2.87031E-04 0.11689  7.73101E-05 0.23409 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.54594E-01 0.09084  1.24891E-02 0.00090  3.22194E-02 0.00090  1.06458E-01 0.00288  3.01394E-01 0.00201  1.26325E+00 0.00711  7.63205E+00 0.07164 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93338E-03 0.04055  2.09456E-04 0.15854  6.40615E-04 0.07412  5.14551E-04 0.09013  1.20758E-03 0.06254  2.84019E-04 0.11713  7.71551E-05 0.23356 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.50577E-01 0.09029  1.24891E-02 0.00090  3.22174E-02 0.00090  1.06452E-01 0.00286  3.01411E-01 0.00200  1.26341E+00 0.00711  7.71524E+00 0.06934 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.84028E+03 0.03734 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.71699E-07 0.00081 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.86504E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40032E-03 0.00523 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90105E+03 0.00520 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.32547E-09 0.00106 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.81018E-05 0.00492  6.81104E-05 0.00495  2.23394E-06 0.27919 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.38471E-05 0.01781  6.38276E-05 0.01777  2.51179E-06 0.48769 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.58816E-04 0.01643  3.58931E-04 0.01649  3.33688E-04 0.27454 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61783E+01 0.01127 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48723E+01 0.00029  3.90664E+01 0.00039 ];

