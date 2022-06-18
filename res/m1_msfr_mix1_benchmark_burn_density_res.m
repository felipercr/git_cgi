
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 21:31:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01160E+00  9.98391E-01  1.00091E+00  9.97661E-01  9.89701E-01  9.97499E-01  1.00072E+00  1.00162E+00  1.00285E+00  1.00385E+00  1.00424E+00  1.00053E+00  9.99424E-01  1.00193E+00  9.92528E-01  1.00231E+00  9.91048E-01  1.00319E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.35385E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56461E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30992E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45449E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77543E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48546E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47569E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.48053E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.06427E+00 0.00041  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334001 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33419E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33419E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.30714E+02 ;
RUNNING_TIME              (idx, 1)        =  3.44503E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.83483E-01  1.83483E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.29769E+01  3.29769E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.75850E-01  2.19967E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.42686E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.59974 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98620E+00 3.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.57954E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  2.68412E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.11084E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.05975E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.68412E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.11084E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.57982E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.36945E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.57982E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.36945E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.24239E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.05880E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09506E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.23048E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15175E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.77844E-01 0.00080 ];
TH232_FISS                (idx, [1:   4]) = [  1.55594E+18 0.00386  1.65868E-02 0.00387 ];
U233_FISS                 (idx, [1:   4]) = [  9.22529E+19 0.00049  9.83413E-01 6.5E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27658E+19 0.00060  7.48647E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34206E+19 0.00136  1.08312E-01 0.00135 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002557 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16494E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002557 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5136394 5.14120E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3888775 3.89227E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 977388 9.78176E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002557 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.42144E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36456E+20 3.9E-06  2.36456E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37769E+19 1.2E-07  9.37769E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23928E+20 0.00031  1.09322E+20 0.00024  1.46059E+19 0.00169 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17705E+20 0.00018  2.03099E+20 0.00013  1.46059E+19 0.00169 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41016E+20 0.00036  2.41016E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32468E+22 0.00025  3.14670E+22 0.00024  1.77982E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35762E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41281E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07990E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29423E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.26740E-01 0.07350 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.86578E-01 0.00791 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.30821E-04 0.01469 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.08137E+04 0.02606 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02193E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99988E-01 1.2E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.11415E-01 0.06142 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.61440E-01 0.06142 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52147E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.81429E-01 0.00043  1.08698E-01 0.00042  3.50252E-04 0.00865 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.81148E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.81117E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.81148E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08753E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03021E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02997E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32744E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32761E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49531E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49481E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28887E-03 0.00582  2.68710E-04 0.01782  8.08119E-04 0.01132  6.01185E-04 0.01320  1.28910E-03 0.00892  2.69447E-04 0.02042  5.23058E-05 0.04637 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.49596E-01 0.01606  1.24751E-02 3.0E-05  3.23385E-02 9.9E-05  1.06021E-01 0.00057  2.97438E-01 0.00026  1.23548E+00 0.00022  5.28511E+00 0.03954 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20679E-03 0.00916  2.52735E-04 0.02716  8.07584E-04 0.01663  5.81523E-04 0.01913  1.25192E-03 0.01396  2.62417E-04 0.03192  5.06101E-05 0.06820 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46066E-01 0.02495  1.24757E-02 3.9E-05  3.23384E-02 0.00014  1.06052E-01 0.00082  2.97506E-01 0.00039  1.23512E+00 0.00030  6.48165E+00 0.02993 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.85544E-07 0.00115  8.85261E-07 0.00115  9.68423E-07 0.01784 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.69058E-07 0.00109  8.68779E-07 0.00109  9.50517E-07 0.01786 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21339E-03 0.00871  2.57931E-04 0.03153  7.91165E-04 0.01705  5.92128E-04 0.02128  1.25952E-03 0.01471  2.64548E-04 0.03174  4.80975E-05 0.07081 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.42009E-01 0.02451  1.24761E-02 4.2E-05  3.23358E-02 0.00015  1.06040E-01 0.00091  2.97267E-01 0.00044  1.23507E+00 0.00041  6.41920E+00 0.04109 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.87220E-07 0.01963  7.86894E-07 0.01963  8.62899E-07 0.05184 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.72860E-07 0.01963  7.72539E-07 0.01963  8.46890E-07 0.05176 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98556E-03 0.04362  2.17009E-04 0.13476  7.74990E-04 0.08107  4.47033E-04 0.09205  1.24684E-03 0.06401  2.56180E-04 0.12305  4.35150E-05 0.32955 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52241E-01 0.11326  1.24738E-02 0.00017  3.23340E-02 0.00050  1.06109E-01 0.00340  2.97208E-01 0.00163  1.23450E+00 0.00119  6.26138E+00 0.12253 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99956E-03 0.04335  2.19000E-04 0.12917  7.99246E-04 0.07927  4.36236E-04 0.08908  1.24896E-03 0.06353  2.50044E-04 0.12094  4.60785E-05 0.31266 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.58374E-01 0.10885  1.24741E-02 0.00016  3.23358E-02 0.00050  1.06098E-01 0.00335  2.97220E-01 0.00162  1.23452E+00 0.00119  6.26138E+00 0.12253 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81651E+03 0.03958 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.78809E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.62438E-07 0.00048 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24752E-03 0.00523 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.69556E+03 0.00522 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.22984E-09 0.00126 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.22119E-05 0.00405  7.22015E-05 0.00406  6.08011E-06 0.20129 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.68572E-05 0.01463  7.69573E-05 0.01473  5.10458E-06 0.25879 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.07017E-04 0.01407  5.06184E-04 0.01414  7.62567E-04 0.19241 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74414E+01 0.01175 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47569E+01 0.00025  4.03622E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 22:20:30 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01098E+00  9.99988E-01  9.98342E-01  1.00422E+00  9.93198E-01  9.94893E-01  9.94687E-01  9.98464E-01  1.00539E+00  1.00158E+00  9.98200E-01  1.00077E+00  9.99816E-01  1.00336E+00  9.94643E-01  1.00204E+00  9.96127E-01  1.00331E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.34382E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56562E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30873E-01 0.00020  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45303E-01 0.00019  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77637E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49301E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48323E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.50028E+01 0.00033  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.06263E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333483 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33378E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33378E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.20065E+02 ;
RUNNING_TIME              (idx, 1)        =  8.38343E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.04100E-01  1.20617E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  8.22156E+01  4.92387E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.78667E-02  1.78667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.42183E-01  3.66283E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.35069E+01  5.14007E+02 ];
CPU_USAGE                 (idx, 1)        = 9.78197 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98957E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77030E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  5.77944E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60428E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.66893E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.48554E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.04702E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26281E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46839E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.45948E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65725E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.39730E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.80266E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06217E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17699E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44219E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35376E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17372E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55143E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60144E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38113E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39819E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51668E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.53285E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.31274E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.30217E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26594E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.11361E-04 -2.39620E+25  2.15199E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.90589E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.58044E+18 0.00377  1.68636E-02 0.00372 ];
U233_FISS                 (idx, [1:   4]) = [  9.21099E+19 0.00055  9.82849E-01 6.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.16054E+14 0.27176  3.37762E-06 0.27176 ];
TH232_CAPT                (idx, [1:   4]) = [  9.39060E+19 0.00053  7.44478E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34022E+19 0.00127  1.06253E-01 0.00122 ];
U235_CAPT                 (idx, [1:   4]) = [  1.69902E+14 0.37418  1.34497E-06 0.37420 ];
SM149_CAPT                (idx, [1:   4]) = [  1.39909E+16 0.04123  1.10984E-04 0.04129 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001353 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15623E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001353 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5176897 5.18220E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3846288 3.85027E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 978168 9.79089E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001353 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.41561E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36447E+20 3.8E-06  2.36447E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.3E-07  9.37766E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26178E+20 0.00033  1.11419E+20 0.00025  1.47585E+19 0.00194 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19955E+20 0.00019  2.05196E+20 0.00013  1.47585E+19 0.00194 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43406E+20 0.00037  2.43406E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36207E+22 0.00025  3.18231E+22 0.00023  1.79754E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.38323E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43787E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09244E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29514E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29514E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.69508E-01 0.06667 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.97245E-01 0.00807 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.69945E-04 0.01458 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.03861E+04 0.02624 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02100E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.66731E-01 0.05483 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.11248E-01 0.05483 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52139E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.71010E-01 0.00044  1.07522E-01 0.00044  3.48008E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.71025E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.71452E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.71025E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07637E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03392E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03391E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32252E-02 0.00126 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32240E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50851E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49917E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34742E-03 0.00564  2.74029E-04 0.01999  8.31142E-04 0.01147  6.26179E-04 0.01281  1.29360E-03 0.00875  2.70084E-04 0.01784  5.23881E-05 0.04540 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.44794E-01 0.01449  1.24757E-02 2.9E-05  3.23381E-02 9.4E-05  1.05883E-01 0.00056  2.97386E-01 0.00026  1.23577E+00 0.00023  5.02942E+00 0.04015 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27402E-03 0.00783  2.65119E-04 0.02922  8.13304E-04 0.01694  6.24126E-04 0.01904  1.26675E-03 0.01273  2.53493E-04 0.02916  5.12225E-05 0.06517 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.41536E-01 0.02289  1.24756E-02 3.9E-05  3.23369E-02 0.00014  1.05851E-01 0.00077  2.97393E-01 0.00039  1.23614E+00 0.00029  6.36688E+00 0.02956 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.96983E-07 0.00125  8.96617E-07 0.00125  1.01470E-06 0.01734 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.70915E-07 0.00112  8.70560E-07 0.00112  9.85173E-07 0.01731 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21556E-03 0.00916  2.61922E-04 0.03088  8.01870E-04 0.01938  5.98564E-04 0.02122  1.23883E-03 0.01440  2.62440E-04 0.03329  5.19337E-05 0.07496 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.52421E-01 0.02597  1.24756E-02 5.6E-05  3.23374E-02 0.00016  1.05873E-01 0.00096  2.97332E-01 0.00046  1.23574E+00 0.00041  6.78274E+00 0.03868 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.97668E-07 0.01957  7.97449E-07 0.01957  8.45665E-07 0.04912 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.74871E-07 0.01956  7.74658E-07 0.01956  8.21554E-07 0.04914 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.79743E-03 0.04449  2.67343E-04 0.12873  7.73687E-04 0.07291  5.27995E-04 0.09318  9.85284E-04 0.07072  1.99223E-04 0.13982  4.39021E-05 0.24888 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.16583E-01 0.08625  1.24760E-02 0.00012  3.23408E-02 0.00052  1.05857E-01 0.00306  2.97190E-01 0.00159  1.23511E+00 0.00123  6.50900E+00 0.10223 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81117E-03 0.04396  2.66525E-04 0.12591  7.72523E-04 0.06919  5.30929E-04 0.09261  9.91854E-04 0.06903  2.01668E-04 0.13814  4.76760E-05 0.23560 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.29205E-01 0.09149  1.24760E-02 0.00012  3.23409E-02 0.00052  1.05829E-01 0.00299  2.97215E-01 0.00160  1.23514E+00 0.00122  6.50900E+00 0.10223 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.51845E+03 0.04001 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.90986E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.65098E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23009E-03 0.00613 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.62590E+03 0.00618 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17746E-09 0.00118 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.15343E-05 0.00433  7.15471E-05 0.00435  3.13438E-06 0.26932 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.49795E-05 0.01562  7.48861E-05 0.01568  4.81829E-06 0.36494 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.50470E-04 0.01439  4.50582E-04 0.01443  4.16719E-04 0.26325 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74831E+01 0.01159 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48323E+01 0.00029  4.04326E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 23:12:33 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00586E+00  1.00143E+00  9.98040E-01  1.00171E+00  9.92495E-01  1.00317E+00  9.99631E-01  1.00499E+00  1.00194E+00  9.96693E-01  9.96193E-01  9.95684E-01  9.98093E-01  1.00760E+00  9.98358E-01  1.00200E+00  9.93494E-01  1.00261E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.31582E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56842E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31193E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45533E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77276E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50749E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49773E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.51911E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.04785E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333728 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33380E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33380E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.33565E+03 ;
RUNNING_TIME              (idx, 1)        =  1.35888E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.33200E-01  1.29100E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.34110E+02  5.18945E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.75667E-02  1.97000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.05115E+00  4.08917E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.35518E+02  7.28168E+02 ];
CPU_USAGE                 (idx, 1)        = 9.82901 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98980E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.81977E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.40137E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65532E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.26631E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.87398E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31083E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49601E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49277E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.77436E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.40447E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.87838E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.04416E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89575E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60005E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49285E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34383E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33197E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54087E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.69725E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17646E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.01886E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51649E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.36471E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.54041E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.39885E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61906E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.92596E-04 -1.27512E+26  2.15302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.07873E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59657E+18 0.00393  1.70379E-02 0.00387 ];
U233_FISS                 (idx, [1:   4]) = [  9.19715E+19 0.00052  9.81519E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.14416E+16 0.04561  1.22099E-04 0.04565 ];
TH232_CAPT                (idx, [1:   4]) = [  9.53453E+19 0.00052  7.38382E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34106E+19 0.00126  1.03857E-01 0.00121 ];
U235_CAPT                 (idx, [1:   4]) = [  3.87489E+15 0.08105  2.99775E-05 0.08102 ];
SM149_CAPT                (idx, [1:   4]) = [  6.34515E+16 0.01961  4.91349E-04 0.01959 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001401 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.17297E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001401 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5230138 5.23573E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3795581 3.79939E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 975682 9.76611E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001401 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.10479E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36435E+20 3.6E-06  2.36435E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37760E+19 1.2E-07  9.37760E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29133E+20 0.00031  1.14093E+20 0.00022  1.50400E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22909E+20 0.00018  2.07869E+20 0.00012  1.50400E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46628E+20 0.00035  2.46628E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.41089E+22 0.00026  3.22771E+22 0.00023  1.83186E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40866E+19 0.00131 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46996E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11049E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29739E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29739E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.92045E-01 0.04808 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.29758E-01 0.00922 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.00233E-04 0.01836 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.54617E+03 0.02889 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02343E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.17452E-01 0.03161 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.37620E-01 0.03161 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52127E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.57988E-01 0.00043  1.06094E-01 0.00042  3.40995E-04 0.00887 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.58366E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.58703E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.58366E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06205E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04270E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04313E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31100E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31026E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50839E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50681E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38012E-03 0.00582  2.72327E-04 0.01980  8.28832E-04 0.01167  6.21365E-04 0.01305  1.32343E-03 0.00868  2.80378E-04 0.01916  5.37859E-05 0.04242 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47915E-01 0.01512  1.24754E-02 2.9E-05  3.23354E-02 9.0E-05  1.05968E-01 0.00055  2.97504E-01 0.00028  1.23564E+00 0.00022  5.24436E+00 0.03756 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25123E-03 0.00859  2.69361E-04 0.02999  7.89127E-04 0.01784  6.01404E-04 0.02082  1.26825E-03 0.01302  2.65467E-04 0.03001  5.76247E-05 0.06453 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56534E-01 0.02429  1.24752E-02 4.2E-05  3.23298E-02 0.00012  1.06016E-01 0.00083  2.97487E-01 0.00039  1.23571E+00 0.00030  6.23181E+00 0.02978 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.14412E-07 0.00124  9.14138E-07 0.00126  1.00416E-06 0.01707 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.75948E-07 0.00117  8.75684E-07 0.00118  9.62213E-07 0.01714 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20847E-03 0.00899  2.66583E-04 0.03197  7.61660E-04 0.01825  6.08441E-04 0.02127  1.25982E-03 0.01423  2.63777E-04 0.03316  4.81961E-05 0.07311 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.42172E-01 0.02463  1.24751E-02 5.1E-05  3.23389E-02 0.00016  1.05924E-01 0.00090  2.97665E-01 0.00046  1.23560E+00 0.00041  6.03923E+00 0.04428 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.10383E-07 0.01963  8.09958E-07 0.01963  9.79643E-07 0.06705 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.76521E-07 0.01963  7.76115E-07 0.01963  9.38121E-07 0.06696 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93796E-03 0.04447  2.97094E-04 0.12999  6.50827E-04 0.08249  5.25890E-04 0.09097  1.20471E-03 0.06352  2.31991E-04 0.14441  2.74515E-05 0.31783 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.12696E-01 0.07914  1.24763E-02 0.00012  3.23409E-02 0.00062  1.06230E-01 0.00334  2.97480E-01 0.00159  1.23360E+00 0.00126  6.32329E+00 0.14044 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.95045E-03 0.04381  2.95400E-04 0.12954  6.44007E-04 0.07950  5.32095E-04 0.08996  1.20986E-03 0.06299  2.38352E-04 0.13916  3.07343E-05 0.31135 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.24111E-01 0.08267  1.24764E-02 0.00011  3.23404E-02 0.00062  1.06209E-01 0.00330  2.97504E-01 0.00158  1.23352E+00 0.00126  6.32329E+00 0.14044 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65521E+03 0.04073 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.07435E-07 0.00059 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.69271E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20564E-03 0.00583 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.53322E+03 0.00589 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99074E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.94564E-05 0.00530  6.94621E-05 0.00531  1.71555E-06 0.38173 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.72710E-05 0.02070  6.73129E-05 0.02077  1.62463E-06 0.48811 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.88203E-04 0.01761  2.88479E-04 0.01760  2.03847E-04 0.37483 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72542E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49773E+01 0.00026  4.06216E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 00:05:20 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00688E+00  1.00709E+00  1.00795E+00  9.95511E-01  9.94110E-01  9.92762E-01  9.99549E-01  1.00379E+00  1.00092E+00  1.01089E+00  9.98481E-01  1.00672E+00  1.00363E+00  1.00185E+00  9.96486E-01  9.91865E-01  9.86322E-01  9.95202E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.30052E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56995E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31386E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45681E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76428E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51819E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50843E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.53379E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.04095E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333388 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33385E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33385E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.85884E+03 ;
RUNNING_TIME              (idx, 1)        =  1.88670E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.68317E-01  1.35117E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.86725E+02  5.26151E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.58833E-02  1.83167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.42022E+00  3.68983E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.88342E+02  7.60869E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85230 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98987E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84433E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.51929E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67135E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.28481E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91161E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33617E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57595E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50595E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.83274E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.56651E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.03724E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.23860E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79438E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74262E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48798E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36099E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34515E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55856E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39732E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42545E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09247E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54636E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.30449E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70210E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.47547E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.15881E-03 -2.49346E+26  2.15424E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.20960E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.59928E+18 0.00384  1.70385E-02 0.00377 ];
U233_FISS                 (idx, [1:   4]) = [  9.19831E+19 0.00047  9.80033E-01 7.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.52008E+16 0.02460  4.81642E-04 0.02462 ];
TH232_CAPT                (idx, [1:   4]) = [  9.64382E+19 0.00053  7.34790E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34077E+19 0.00130  1.02158E-01 0.00125 ];
U235_CAPT                 (idx, [1:   4]) = [  1.65863E+16 0.03982  1.26446E-04 0.03980 ];
SM149_CAPT                (idx, [1:   4]) = [  1.03214E+17 0.01506  7.86580E-04 0.01510 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001541 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13961E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001541 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5261714 5.26707E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3763004 3.76664E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 976823 9.77682E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001541 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.21541E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36429E+20 3.5E-06  2.36429E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.2E-07  9.37752E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31212E+20 0.00031  1.15957E+20 0.00022  1.52558E+19 0.00171 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.24988E+20 0.00018  2.09732E+20 0.00012  1.52558E+19 0.00171 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49182E+20 0.00034  2.49182E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44938E+22 0.00025  3.26375E+22 0.00022  1.85624E+21 0.00163 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43629E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.49350E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12464E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29706E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29706E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.43661E-01 0.05031 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06701E-01 0.01208 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.27236E-04 0.02106 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.13110E+03 0.02779 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02235E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.54345E-01 0.03637 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.80649E-01 0.03637 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52123E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.49704E-01 0.00041  1.05176E-01 0.00041  3.41450E-04 0.00953 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.49280E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.48852E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.49280E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05220E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04349E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04250E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30996E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31109E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51682E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52101E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.42504E-03 0.00546  2.73258E-04 0.02037  8.45097E-04 0.01118  6.28238E-04 0.01309  1.33507E-03 0.00904  2.83443E-04 0.01912  5.99291E-05 0.04421 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.64938E-01 0.01608  1.24755E-02 2.8E-05  3.23362E-02 9.1E-05  1.06068E-01 0.00060  2.97483E-01 0.00026  1.23542E+00 0.00023  5.60495E+00 0.03588 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22547E-03 0.00770  2.56148E-04 0.02954  8.11169E-04 0.01604  5.81550E-04 0.01953  1.26006E-03 0.01324  2.62179E-04 0.02842  5.43576E-05 0.06385 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.59243E-01 0.02290  1.24756E-02 3.8E-05  3.23413E-02 0.00015  1.06087E-01 0.00089  2.97509E-01 0.00039  1.23562E+00 0.00029  6.74881E+00 0.02740 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.21757E-07 0.00123  9.21312E-07 0.00123  1.05498E-06 0.02574 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.75360E-07 0.00119  8.74937E-07 0.00118  1.00188E-06 0.02568 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23696E-03 0.00972  2.59925E-04 0.03148  8.03644E-04 0.01893  5.77493E-04 0.02183  1.26948E-03 0.01638  2.69815E-04 0.03121  5.65934E-05 0.07330 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.68698E-01 0.02895  1.24757E-02 4.9E-05  3.23352E-02 0.00016  1.05988E-01 0.00097  2.97222E-01 0.00043  1.23519E+00 0.00042  6.73560E+00 0.03881 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.12866E-07 0.01958  8.12446E-07 0.01958  9.17938E-07 0.05568 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.72056E-07 0.01958  7.71656E-07 0.01958  8.71930E-07 0.05570 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.76214E-03 0.04323  2.33299E-04 0.15016  6.08002E-04 0.07964  5.40201E-04 0.09501  1.08574E-03 0.06726  2.46890E-04 0.11873  4.80126E-05 0.31762 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.89312E-01 0.10576  1.24765E-02 0.00013  3.23413E-02 0.00057  1.06248E-01 0.00322  2.97230E-01 0.00163  1.23624E+00 0.00102  6.38391E+00 0.11738 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.74665E-03 0.04296  2.23594E-04 0.14619  6.03963E-04 0.07803  5.33289E-04 0.09252  1.09289E-03 0.06718  2.45814E-04 0.11761  4.70936E-05 0.30784 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.86546E-01 0.10021  1.24765E-02 0.00013  3.23417E-02 0.00057  1.06238E-01 0.00321  2.97262E-01 0.00165  1.23629E+00 0.00101  6.33063E+00 0.11744 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.40956E+03 0.03886 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.14341E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.68306E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29242E-03 0.00655 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.60036E+03 0.00644 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91239E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.93848E-05 0.00644  6.93586E-05 0.00645  1.74504E-06 0.37992 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.29329E-05 0.02395  6.26661E-05 0.02426  2.86609E-06 0.41307 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.17838E-04 0.02079  2.17884E-04 0.02084  1.98725E-04 0.37500 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71292E+01 0.01170 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50843E+01 0.00027  4.07202E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 00:59:43 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01232E+00  1.00004E+00  9.94292E-01  9.98065E-01  9.95698E-01  1.00112E+00  9.96203E-01  1.00046E+00  9.94478E-01  9.99603E-01  9.96139E-01  1.00057E+00  1.00211E+00  1.00160E+00  9.97286E-01  1.00152E+00  1.00167E+00  1.00683E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.28646E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57135E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31066E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45338E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76132E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53102E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52123E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57100E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.03880E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333743 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33381E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33381E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.39699E+03 ;
RUNNING_TIME              (idx, 1)        =  2.43049E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.11733E-01  1.43417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.40926E+02  5.42011E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.39167E-02  1.80333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.89352E+00  4.73250E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.42622E+02  7.70387E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86215 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98912E+00 2.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.85468E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.54801E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66623E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.67105E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93572E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35131E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58026E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49902E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00276E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.68777E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.32342E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.34717E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69821E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85293E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93615E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32452E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31722E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52011E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34913E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81409E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09123E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52086E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.56868E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.75228E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.57269E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64039E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.35159E-03 -5.06002E+26  2.15681E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.46863E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.62409E+18 0.00397  1.73255E-02 0.00388 ];
U233_FISS                 (idx, [1:   4]) = [  9.14892E+19 0.00051  9.76046E-01 8.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.78676E+17 0.01174  1.90623E-03 0.01175 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81542E+19 0.00054  7.31162E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33534E+19 0.00141  9.94718E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  6.13478E+16 0.02132  4.56981E-04 0.02134 ];
SM149_CAPT                (idx, [1:   4]) = [  1.45238E+17 0.01285  1.08199E-03 0.01287 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001430 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14537E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001430 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5312763 5.31826E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3709801 3.71342E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 978866 9.79781E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001430 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.86265E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36415E+20 3.7E-06  2.36415E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34202E+20 0.00030  1.18812E+20 0.00022  1.53901E+19 0.00170 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27975E+20 0.00017  2.12585E+20 0.00012  1.53901E+19 0.00170 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.52423E+20 0.00036  2.52423E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50163E+22 0.00026  3.31503E+22 0.00023  1.86594E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.47325E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.52708E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14250E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29340E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29340E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.72296E-01 0.05212 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17608E-01 0.01266 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.95944E-04 0.02402 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.80114E+03 0.02831 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02025E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.54369E-01 0.03547 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.80463E-01 0.03547 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52115E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.36158E-01 0.00043  1.03681E-01 0.00042  3.41200E-04 0.00966 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.36619E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.36619E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.36619E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03836E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04482E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04526E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30821E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30748E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54235E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54369E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49512E-03 0.00589  2.82119E-04 0.02003  8.59114E-04 0.01108  6.43829E-04 0.01275  1.37046E-03 0.00917  2.85486E-04 0.02195  5.41112E-05 0.04494 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42866E-01 0.01461  1.24759E-02 2.5E-05  3.23264E-02 9.5E-05  1.06225E-01 0.00063  2.97537E-01 0.00026  1.23633E+00 0.00032  5.15612E+00 0.03841 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25604E-03 0.00812  2.56740E-04 0.02999  7.92821E-04 0.01620  6.00311E-04 0.01876  1.28184E-03 0.01300  2.71403E-04 0.02932  5.29275E-05 0.06193 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48608E-01 0.02036  1.24764E-02 3.3E-05  3.23217E-02 0.00012  1.06304E-01 0.00088  2.97648E-01 0.00039  1.23627E+00 0.00041  6.29961E+00 0.02938 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.38985E-07 0.00122  9.38615E-07 0.00123  1.04743E-06 0.01707 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.78996E-07 0.00117  8.78649E-07 0.00117  9.80458E-07 0.01706 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28246E-03 0.00982  2.73572E-04 0.03308  8.04859E-04 0.01748  5.89491E-04 0.02110  1.28745E-03 0.01539  2.76441E-04 0.03163  5.06472E-05 0.07005 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53212E-01 0.02505  1.24761E-02 4.4E-05  3.23282E-02 0.00019  1.06278E-01 0.00112  2.97707E-01 0.00046  1.23635E+00 0.00048  6.74209E+00 0.03867 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.30296E-07 0.01961  8.29832E-07 0.01961  9.55892E-07 0.06643 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.77370E-07 0.01960  7.76935E-07 0.01961  8.95427E-07 0.06653 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.95797E-03 0.04457  2.99796E-04 0.13123  7.31082E-04 0.07717  5.08553E-04 0.10661  1.16010E-03 0.06668  2.23000E-04 0.13762  3.54360E-05 0.28731 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.21832E-01 0.07727  1.24771E-02 0.00010  3.23315E-02 0.00053  1.05960E-01 0.00337  2.98545E-01 0.00196  1.23669E+00 0.00170  5.45663E+00 0.15206 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.96481E-03 0.04426  2.90043E-04 0.13225  7.29604E-04 0.07695  5.08817E-04 0.10533  1.17623E-03 0.06533  2.27214E-04 0.13328  3.29014E-05 0.28319 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.21912E-01 0.07769  1.24771E-02 0.00010  3.23330E-02 0.00054  1.05934E-01 0.00333  2.98573E-01 0.00196  1.23651E+00 0.00171  5.45663E+00 0.15206 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.57966E+03 0.04052 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.32469E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.72893E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24554E-03 0.00648 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.48089E+03 0.00649 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89988E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.81456E-05 0.00687  6.81286E-05 0.00688  2.20920E-06 0.34078 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.12517E-05 0.02808  6.12727E-05 0.02808  1.38929E-06 0.51281 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.88942E-04 0.02308  1.88707E-04 0.02325  2.60875E-04 0.33040 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73488E+01 0.01164 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52123E+01 0.00027  4.08089E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 01:54:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01231E+00  9.95515E-01  9.96681E-01  9.95265E-01  9.89493E-01  9.99910E-01  9.97416E-01  9.97068E-01  1.00344E+00  1.00167E+00  1.00551E+00  1.00314E+00  1.00317E+00  9.99106E-01  9.96544E-01  1.00594E+00  9.91169E-01  1.00665E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.28204E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57180E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30976E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45235E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75839E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54360E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53380E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59874E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.04469E+00 0.00043  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334073 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33362E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33362E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.93720E+03 ;
RUNNING_TIME              (idx, 1)        =  2.98149E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.49317E-01  1.37583E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.95849E+02  5.49229E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.32833E-02  1.93667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.92573E+00  1.03217E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.97154E+02  7.86376E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85145 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98900E+00 2.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84271E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.61859E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67636E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.20448E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97214E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37539E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61395E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50628E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10193E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79927E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.74929E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.51413E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25466E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94756E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35117E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33556E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32528E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53108E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02017E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17233E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13241E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55528E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.46994E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85704E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.65368E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96058E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.52603E-03 -7.58711E+26  2.15933E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.65232E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.62919E+18 0.00383  1.73654E-02 0.00377 ];
U233_FISS                 (idx, [1:   4]) = [  9.11656E+19 0.00055  9.71756E-01 8.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.74013E+17 0.00841  3.98713E-03 0.00844 ];
TH232_CAPT                (idx, [1:   4]) = [  9.93399E+19 0.00051  7.27270E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32821E+19 0.00141  9.72392E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  1.33610E+17 0.01296  9.78077E-04 0.01292 ];
SM149_CAPT                (idx, [1:   4]) = [  1.63063E+17 0.01248  1.19374E-03 0.01247 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000856 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15736E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000856 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5348063 5.35406E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3673470 3.67728E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 979323 9.80234E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000856 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.90573E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36399E+20 3.7E-06  2.36399E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37698E+19 1.2E-07  9.37698E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36675E+20 0.00033  1.21055E+20 0.00022  1.56203E+19 0.00185 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30445E+20 0.00020  2.14824E+20 0.00012  1.56203E+19 0.00185 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.55123E+20 0.00036  2.55123E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54664E+22 0.00025  3.35734E+22 0.00022  1.89301E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50088E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.55453E+20 0.00023 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15796E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28836E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28836E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88026E-01 0.04926 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06899E-01 0.01442 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.83039E-04 0.02590 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.90539E+03 0.02909 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01980E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.36775E-01 0.03637 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.64538E-01 0.03637 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52106E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.27061E-01 0.00044  1.02676E-01 0.00044  3.31866E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.26495E-01 0.00023 ];
COL_KEFF                  (idx, [1:   2]) = [  9.26646E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.26495E-01 0.00023 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02717E+00 0.00020 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04432E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04510E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30887E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30770E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56786E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.56817E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.55970E-03 0.00567  2.83828E-04 0.01980  8.74360E-04 0.01112  6.48806E-04 0.01287  1.41006E-03 0.00812  2.90651E-04 0.01952  5.19962E-05 0.04398 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43190E-01 0.01447  1.24756E-02 2.8E-05  3.23231E-02 9.7E-05  1.06171E-01 0.00060  2.97728E-01 0.00026  1.23663E+00 0.00032  5.30618E+00 0.03914 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28724E-03 0.00832  2.60892E-04 0.02795  8.06743E-04 0.01570  5.87773E-04 0.01853  1.31728E-03 0.01221  2.64541E-04 0.02784  5.00178E-05 0.06965 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45735E-01 0.02147  1.24756E-02 4.2E-05  3.23239E-02 0.00015  1.06251E-01 0.00089  2.97729E-01 0.00038  1.23580E+00 0.00040  6.61806E+00 0.02903 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.49723E-07 0.00121  9.49343E-07 0.00121  1.05848E-06 0.02566 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.80395E-07 0.00111  8.80041E-07 0.00111  9.81714E-07 0.02596 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22735E-03 0.00895  2.52647E-04 0.03190  7.96603E-04 0.01837  5.80335E-04 0.02199  1.30204E-03 0.01365  2.47990E-04 0.03217  4.77359E-05 0.07764 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.42025E-01 0.02521  1.24765E-02 4.5E-05  3.23234E-02 0.00016  1.06038E-01 0.00097  2.97514E-01 0.00043  1.23630E+00 0.00058  6.79207E+00 0.04105 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.35902E-07 0.01964  8.35762E-07 0.01964  8.27991E-07 0.05040 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.74938E-07 0.01963  7.74807E-07 0.01963  7.67949E-07 0.05044 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.86314E-03 0.04449  2.12623E-04 0.15210  7.53457E-04 0.07870  5.55631E-04 0.09461  1.12475E-03 0.06716  1.79346E-04 0.17017  3.73334E-05 0.35435 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.23853E-01 0.10093  1.24762E-02 0.00014  3.23291E-02 0.00055  1.06149E-01 0.00345  2.97904E-01 0.00173  1.23498E+00 0.00126  8.12896E+00 0.10532 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85082E-03 0.04408  2.20002E-04 0.14503  7.48386E-04 0.07795  5.45385E-04 0.09328  1.12201E-03 0.06648  1.81464E-04 0.15828  3.35734E-05 0.34538 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.18414E-01 0.09704  1.24762E-02 0.00014  3.23274E-02 0.00053  1.06138E-01 0.00341  2.97779E-01 0.00169  1.23504E+00 0.00126  8.12896E+00 0.10532 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.44384E+03 0.04040 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.42695E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.73882E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23108E-03 0.00576 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.42812E+03 0.00581 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89759E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63696E-05 0.00969  6.63403E-05 0.00965  1.76909E-06 0.39999 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.52975E-05 0.02593  5.52414E-05 0.02595  1.46351E-06 0.47012 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77780E-04 0.02499  1.77709E-04 0.02503  2.00866E-04 0.37457 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68921E+01 0.01125 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53380E+01 0.00030  4.08759E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 02:50:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01012E+00  1.00198E+00  1.00414E+00  9.99439E-01  9.92815E-01  9.97974E-01  9.98508E-01  1.00046E+00  1.00153E+00  9.95994E-01  1.00040E+00  9.94236E-01  9.98846E-01  9.96769E-01  9.97464E-01  1.00549E+00  1.00004E+00  1.00379E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.28152E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57185E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30427E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44698E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76049E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55446E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54466E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.63986E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.05571E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334355 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33399E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33399E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.48406E+03 ;
RUNNING_TIME              (idx, 1)        =  3.54187E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.99217E-01  1.49900E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.51697E+02  5.58483E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.12117E-01  1.88333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.26492E+00  1.33913E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.52891E+02  7.93624E+02 ];
CPU_USAGE                 (idx, 1)        = 9.83678 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99002E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.82618E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.64837E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67516E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.27083E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99924E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39147E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61618E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50306E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27758E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.93402E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.78847E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.66779E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93996E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06630E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06901E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30939E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30416E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50262E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.83710E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78548E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13889E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.56853E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.24826E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.90406E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.76641E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60097E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.96199E-03 -1.28287E+27  2.16457E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.97680E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.65921E+18 0.00389  1.77109E-02 0.00379 ];
U233_FISS                 (idx, [1:   4]) = [  9.00874E+19 0.00051  9.61690E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  9.08405E+17 0.00548  9.69747E-03 0.00547 ];
PU239_FISS                (idx, [1:   4]) = [  2.63387E+13 1.00000  2.77408E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01132E+20 0.00054  7.21939E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31744E+19 0.00137  9.40472E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  3.18385E+17 0.00935  2.27296E-03 0.00935 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77321E+17 0.01142  1.26611E-03 0.01148 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001983 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14555E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001983 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5405855 5.41117E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3615153 3.61855E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 980975 9.81736E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001983 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.31323E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36360E+20 3.8E-06  2.36360E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37617E+19 1.1E-07  9.37617E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40024E+20 0.00031  1.24202E+20 0.00023  1.58222E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33786E+20 0.00019  2.17964E+20 0.00013  1.58222E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58880E+20 0.00037  2.58880E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.60887E+22 0.00026  3.41796E+22 0.00023  1.90905E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.54162E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59202E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17780E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27823E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27823E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.52548E-01 0.04803 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21823E-01 0.01361 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90459E-04 0.02663 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.17724E+03 0.02980 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01829E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.52020E-01 0.03399 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.78247E-01 0.03399 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52086E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99703E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.12082E-01 0.00044  1.01019E-01 0.00044  3.32416E-04 0.00876 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12934E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.13046E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12934E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01231E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03828E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03812E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31684E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31683E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.61280E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.61315E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.62685E-03 0.00526  2.95355E-04 0.01884  8.77884E-04 0.01071  6.71393E-04 0.01220  1.41437E-03 0.00843  3.06608E-04 0.01931  6.12334E-05 0.04389 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.59070E-01 0.01464  1.24340E-02 0.00334  3.23181E-02 0.00010  1.06212E-01 0.00056  2.98112E-01 0.00027  1.23969E+00 0.00039  5.41753E+00 0.03584 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27157E-03 0.00790  2.73980E-04 0.02905  7.87528E-04 0.01602  6.04463E-04 0.01830  1.26726E-03 0.01257  2.78525E-04 0.02922  5.98083E-05 0.05815 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.69032E-01 0.02250  1.24759E-02 4.0E-05  3.23147E-02 0.00015  1.06250E-01 0.00082  2.98139E-01 0.00040  1.23963E+00 0.00050  6.55955E+00 0.02743 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.67919E-07 0.00118  9.67575E-07 0.00119  1.07244E-06 0.01798 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.82775E-07 0.00111  8.82460E-07 0.00111  9.78414E-07 0.01807 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27719E-03 0.00889  2.62156E-04 0.03225  7.89057E-04 0.01802  5.97991E-04 0.02127  1.29847E-03 0.01447  2.76837E-04 0.03135  5.26794E-05 0.07359 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.59426E-01 0.02409  1.24762E-02 4.9E-05  3.23110E-02 0.00018  1.06018E-01 0.00091  2.98254E-01 0.00048  1.23923E+00 0.00070  6.84060E+00 0.03879 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.56911E-07 0.01964  8.56239E-07 0.01964  1.01861E-06 0.05861 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.81719E-07 0.01963  7.81106E-07 0.01963  9.28952E-07 0.05861 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92042E-03 0.04259  2.71313E-04 0.14201  6.59445E-04 0.08252  5.12944E-04 0.09241  1.17645E-03 0.06483  2.34454E-04 0.12960  6.58223E-05 0.27343 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.12707E-01 0.12008  1.24731E-02 0.00020  3.22995E-02 0.00042  1.06473E-01 0.00355  2.97324E-01 0.00156  1.24097E+00 0.00230  6.06414E+00 0.12055 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93379E-03 0.04153  2.62038E-04 0.14081  6.65883E-04 0.08119  5.16630E-04 0.09073  1.19475E-03 0.06347  2.31162E-04 0.12724  6.33344E-05 0.27257 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.09684E-01 0.11977  1.24731E-02 0.00020  3.22978E-02 0.00042  1.06483E-01 0.00354  2.97332E-01 0.00155  1.24116E+00 0.00237  6.06414E+00 0.12055 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.41805E+03 0.03824 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.60504E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.76004E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.31511E-03 0.00567 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.45275E+03 0.00584 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92960E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77217E-05 0.00759  6.77228E-05 0.00760  1.16215E-06 0.46289 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.80495E-05 0.02569  5.81642E-05 0.02570  3.12281E-07 0.61593 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.81663E-04 0.02518  1.81823E-04 0.02515  1.30963E-04 0.44565 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70530E+01 0.01139 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54466E+01 0.00028  4.08700E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 03:47:40 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01084E+00  9.91103E-01  9.99639E-01  1.00460E+00  9.97429E-01  9.99639E-01  9.94798E-01  9.97551E-01  1.00323E+00  9.97390E-01  9.95219E-01  1.00491E+00  1.00393E+00  1.00279E+00  9.97213E-01  1.00044E+00  9.94680E-01  1.00461E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.28665E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57134E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30194E-01 0.00020  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44484E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75713E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55568E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54585E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.65034E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.06311E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334040 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33396E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33396E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.03754E+03 ;
RUNNING_TIME              (idx, 1)        =  4.10998E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.14273E+00  1.43517E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.08319E+02  5.66215E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.31533E-01  1.94167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.72238E+00  1.45740E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.09578E+02  8.01656E+02 ];
CPU_USAGE                 (idx, 1)        = 9.82374 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99020E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.81155E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.73411E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69066E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.22322E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02821E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40601E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67236E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51657E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.77528E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.23903E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.34629E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.92266E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.24068E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.34349E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32976E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.31146E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30218E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50181E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.56254E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27997E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.18149E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.68549E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.91348E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99657E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.89652E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.20837E-02 -2.60011E+27  2.17775E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03077E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.66939E+18 0.00400  1.78161E-02 0.00392 ];
U233_FISS                 (idx, [1:   4]) = [  8.77514E+19 0.00053  9.36566E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.48039E+18 0.00313  2.64724E-02 0.00305 ];
PU239_FISS                (idx, [1:   4]) = [  1.02435E+15 0.15809  1.09181E-05 0.15811 ];
PU240_FISS                (idx, [1:   4]) = [  2.54441E+13 1.00000  2.71843E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01804E+20 0.00057  7.07363E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28151E+19 0.00131  8.90444E-02 0.00125 ];
U235_CAPT                 (idx, [1:   4]) = [  8.84308E+17 0.00556  6.14420E-03 0.00552 ];
PU239_CAPT                (idx, [1:   4]) = [  4.46059E+14 0.23598  3.10724E-06 0.23597 ];
PU240_CAPT                (idx, [1:   4]) = [  7.84310E+13 0.57545  5.43983E-07 0.57547 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81405E+17 0.01195  1.26055E-03 0.01196 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001873 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14119E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001873 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5462451 5.46773E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3556245 3.55963E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983177 9.84048E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001873 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -9.31323E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36261E+20 4.0E-06  2.36261E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37385E+19 1.2E-07  9.37385E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43819E+20 0.00030  1.27669E+20 0.00021  1.61499E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37557E+20 0.00018  2.21407E+20 0.00012  1.61499E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.63217E+20 0.00037  2.63217E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67134E+22 0.00025  3.47676E+22 0.00023  1.94576E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59021E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.63459E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19784E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.25205E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.25205E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.40128E-01 0.04913 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.20068E-01 0.01255 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.85940E-04 0.02169 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.76470E+03 0.02701 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01598E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.99703E-01 0.03756 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.30809E-01 0.03756 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52043E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99753E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.97252E-01 0.00043  9.93487E-02 0.00043  3.36359E-04 0.00875 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.97799E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.97627E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.97799E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95812E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00672E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00618E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35906E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35958E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.71857E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.71653E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.81246E-03 0.00558  2.99683E-04 0.01896  9.07818E-04 0.01050  6.88258E-04 0.01211  1.52072E-03 0.00922  3.28686E-04 0.01926  6.72935E-05 0.03870 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.75132E-01 0.01398  1.24753E-02 3.0E-05  3.22948E-02 0.00011  1.06378E-01 0.00061  2.99072E-01 0.00028  1.24585E+00 0.00058  6.11058E+00 0.03030 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.39006E-03 0.00782  2.70478E-04 0.02702  7.98621E-04 0.01644  6.16746E-04 0.01733  1.35177E-03 0.01297  2.93907E-04 0.02691  5.85291E-05 0.06387 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.71083E-01 0.02150  1.24751E-02 4.4E-05  3.22961E-02 0.00016  1.06264E-01 0.00080  2.98947E-01 0.00041  1.24483E+00 0.00071  6.87046E+00 0.02437 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.72690E-07 0.00132  9.72304E-07 0.00132  1.08543E-06 0.01611 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.72701E-07 0.00125  8.72356E-07 0.00125  9.73586E-07 0.01603 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.36270E-03 0.00889  2.78164E-04 0.03077  7.95242E-04 0.01870  5.78857E-04 0.02087  1.35119E-03 0.01447  3.00516E-04 0.02934  5.87308E-05 0.06878 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.77293E-01 0.02342  1.24758E-02 5.1E-05  3.22983E-02 0.00019  1.06383E-01 0.00107  2.99080E-01 0.00050  1.24643E+00 0.00101  6.79378E+00 0.03583 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.59273E-07 0.01960  8.58801E-07 0.01960  9.59599E-07 0.05750 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.70997E-07 0.01960  7.70576E-07 0.01960  8.60342E-07 0.05742 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.28850E-03 0.04274  2.46752E-04 0.14460  7.47434E-04 0.07872  6.07076E-04 0.09599  1.34286E-03 0.06035  2.84393E-04 0.14160  5.99798E-05 0.30635 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.56471E-01 0.09407  1.24747E-02 0.00017  3.23134E-02 0.00066  1.07019E-01 0.00384  2.98798E-01 0.00174  1.24897E+00 0.00286  8.31107E+00 0.08115 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.28059E-03 0.04262  2.49361E-04 0.14041  7.44934E-04 0.07723  6.06244E-04 0.09598  1.33776E-03 0.06017  2.81990E-04 0.14048  6.03016E-05 0.29764 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.57032E-01 0.09378  1.24747E-02 0.00017  3.23113E-02 0.00065  1.07062E-01 0.00386  2.98715E-01 0.00172  1.24890E+00 0.00285  8.11921E+00 0.08753 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.84087E+03 0.03823 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.66013E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.66709E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43962E-03 0.00575 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.56140E+03 0.00584 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92697E-09 0.00071 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79205E-05 0.00934  6.78994E-05 0.00936  2.14293E-06 0.35189 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88645E-05 0.02560  5.87679E-05 0.02564  2.37866E-06 0.47974 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77422E-04 0.02101  1.77192E-04 0.02121  2.32738E-04 0.32965 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.66559E+01 0.01117 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54585E+01 0.00027  4.06332E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 04:44:55 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01158E+00  1.00198E+00  9.97535E-01  1.00164E+00  9.90901E-01  1.00264E+00  1.00036E+00  9.95438E-01  9.99392E-01  1.00159E+00  1.00108E+00  9.98687E-01  9.98133E-01  1.00335E+00  1.00317E+00  1.00362E+00  9.93704E-01  9.95193E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.30221E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56978E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30145E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44474E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75873E+00 0.00027  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53894E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52913E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61889E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.06610E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333949 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33375E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33375E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.59596E+03 ;
RUNNING_TIME              (idx, 1)        =  4.68258E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.29213E+00  1.49400E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.65381E+02  5.70626E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.51050E-01  1.95167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.12365E+00  1.40122E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.66898E+02  8.07640E+02 ];
CPU_USAGE                 (idx, 1)        = 9.81502 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99010E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.80181E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.90001E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73586E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.82098E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03496E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40505E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.83067E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56110E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.22161E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78791E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.13961E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.28750E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04708E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85227E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38322E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38489E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35247E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57393E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.85146E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30712E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28194E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.94424E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.48297E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14318E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.93811E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64364E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.27737E-02 -4.90031E+27  2.20075E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03735E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.62310E+18 0.00431  1.73206E-02 0.00419 ];
U233_FISS                 (idx, [1:   4]) = [  8.43446E+19 0.00054  9.00165E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.90425E+18 0.00215  5.23409E-02 0.00210 ];
PU239_FISS                (idx, [1:   4]) = [  2.50839E+16 0.03352  2.67655E-04 0.03349 ];
PU240_FISS                (idx, [1:   4]) = [  9.50237E+14 0.19139  1.01460E-05 0.19146 ];
PU241_FISS                (idx, [1:   4]) = [  7.68011E+14 0.18336  8.19328E-06 0.18350 ];
TH232_CAPT                (idx, [1:   4]) = [  9.94471E+19 0.00056  6.84857E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23199E+19 0.00149  8.48455E-02 0.00149 ];
U235_CAPT                 (idx, [1:   4]) = [  1.73372E+18 0.00395  1.19396E-02 0.00393 ];
U238_CAPT                 (idx, [1:   4]) = [  2.12533E+14 0.34943  1.45828E-06 0.34942 ];
PU239_CAPT                (idx, [1:   4]) = [  1.27919E+16 0.04459  8.80816E-05 0.04457 ];
PU240_CAPT                (idx, [1:   4]) = [  2.29926E+15 0.11042  1.58315E-05 0.11025 ];
PU241_CAPT                (idx, [1:   4]) = [  7.87469E+13 0.57543  5.44282E-07 0.57542 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84458E+17 0.01196  1.27020E-03 0.01193 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001242 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10517E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001242 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5482232 5.48779E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3537809 3.54113E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981201 9.82127E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001242 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.44007E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36140E+20 4.0E-06  2.36140E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36991E+19 1.1E-07  9.36991E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45266E+20 0.00030  1.29031E+20 0.00022  1.62345E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38965E+20 0.00018  2.22731E+20 0.00013  1.62345E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64604E+20 0.00036  2.64604E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68211E+22 0.00025  3.48682E+22 0.00023  1.95289E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59882E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64953E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19969E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.18619E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.18619E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.13772E-01 0.04602 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27497E-01 0.01176 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94437E-04 0.02327 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.44385E+03 0.02747 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01791E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.65284E-01 0.03131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.90159E-01 0.03131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52020E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99837E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.92448E-01 0.00044  9.88236E-02 0.00043  3.37669E-04 0.00950 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92258E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92463E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92258E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89411E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94548E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94516E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44492E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44512E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.85994E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.85591E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.88992E-03 0.00569  2.94029E-04 0.02015  9.32226E-04 0.01045  6.97978E-04 0.01247  1.53514E-03 0.00883  3.56880E-04 0.01861  7.36621E-05 0.03933 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.86080E-01 0.01445  1.24762E-02 3.0E-05  3.22615E-02 0.00012  1.06526E-01 0.00055  2.99991E-01 0.00030  1.25413E+00 0.00072  5.85251E+00 0.03049 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43891E-03 0.00842  2.55201E-04 0.03068  8.23604E-04 0.01716  6.23495E-04 0.01810  1.36083E-03 0.01306  3.11135E-04 0.02732  6.46466E-05 0.05483 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.88515E-01 0.02128  1.24759E-02 4.0E-05  3.22589E-02 0.00017  1.06570E-01 0.00078  2.99988E-01 0.00045  1.25347E+00 0.00091  6.61750E+00 0.02437 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.52979E-07 0.00126  9.52520E-07 0.00126  1.09127E-06 0.01636 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.50432E-07 0.00118  8.50023E-07 0.00118  9.73840E-07 0.01633 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41917E-03 0.00952  2.45017E-04 0.03436  8.17878E-04 0.01835  6.23034E-04 0.02229  1.36217E-03 0.01486  3.05517E-04 0.03314  6.55538E-05 0.06715 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.83138E-01 0.02432  1.24763E-02 5.6E-05  3.22599E-02 0.00022  1.06485E-01 0.00091  3.00031E-01 0.00054  1.25337E+00 0.00127  6.59644E+00 0.03318 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.38753E-07 0.01963  8.37971E-07 0.01963  1.05157E-06 0.05994 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.48800E-07 0.01962  7.48103E-07 0.01962  9.38834E-07 0.05995 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93727E-03 0.04573  1.76496E-04 0.13561  6.44708E-04 0.08795  5.33931E-04 0.09068  1.22752E-03 0.06597  2.82576E-04 0.12155  7.20401E-05 0.22043 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.44833E-01 0.10840  1.24764E-02 0.00014  3.22474E-02 0.00064  1.06506E-01 0.00317  2.99068E-01 0.00180  1.25352E+00 0.00319  6.62615E+00 0.08633 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93258E-03 0.04606  1.79175E-04 0.13416  6.40306E-04 0.08619  5.36823E-04 0.09168  1.21022E-03 0.06689  2.89684E-04 0.11820  7.63714E-05 0.21678 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.45945E-01 0.10344  1.24763E-02 0.00014  3.22464E-02 0.00063  1.06490E-01 0.00314  2.99124E-01 0.00181  1.25359E+00 0.00319  6.62615E+00 0.08633 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.49953E+03 0.04091 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.42829E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.41379E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.41889E-03 0.00574 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.62646E+03 0.00574 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90656E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.64132E-05 0.00794  6.64347E-05 0.00797  7.57971E-07 0.59651 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.69323E-05 0.02422  5.68750E-05 0.02430  7.13669E-07 0.73784 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.87325E-04 0.02238  1.87757E-04 0.02246  7.64717E-05 0.57715 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.66840E+01 0.01136 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52913E+01 0.00027  4.00838E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 05:42:56 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01293E+00  9.99008E-01  9.98224E-01  1.00645E+00  9.95068E-01  9.98381E-01  9.99292E-01  9.99802E-01  9.99870E-01  1.00058E+00  9.98322E-01  9.97112E-01  9.99116E-01  1.00784E+00  9.90982E-01  9.99762E-01  9.93270E-01  1.00399E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.30940E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56906E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30090E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44432E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75893E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52757E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51776E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59881E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.06655E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333218 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33395E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33395E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.16153E+03 ;
RUNNING_TIME              (idx, 1)        =  5.26269E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.44900E+00  1.56867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.23186E+02  5.78048E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.70367E-01  1.93167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.56332E+00  1.43960E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.24877E+02  8.11071E+02 ];
CPU_USAGE                 (idx, 1)        = 9.80777 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99011E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.79373E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.92396E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74195E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.25989E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02871E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40032E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86071E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56753E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.21298E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.00003E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.03791E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.47892E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13558E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04435E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81938E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38636E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35230E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57435E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.89901E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73707E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29155E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99359E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.91561E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.15555E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.93619E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30374E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.79872E-02 -6.02213E+27  2.21197E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03567E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.60665E+18 0.00422  1.71465E-02 0.00415 ];
U233_FISS                 (idx, [1:   4]) = [  8.32965E+19 0.00060  8.88976E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  5.56067E+18 0.00228  5.93465E-02 0.00222 ];
PU239_FISS                (idx, [1:   4]) = [  5.67059E+16 0.02139  6.04941E-04 0.02134 ];
PU240_FISS                (idx, [1:   4]) = [  2.40944E+15 0.10963  2.56963E-05 0.10978 ];
PU241_FISS                (idx, [1:   4]) = [  2.14261E+15 0.10860  2.28794E-05 0.10863 ];
TH232_CAPT                (idx, [1:   4]) = [  9.82235E+19 0.00053  6.76723E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21421E+19 0.00149  8.36545E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  1.96638E+18 0.00363  1.35479E-02 0.00362 ];
U238_CAPT                 (idx, [1:   4]) = [  4.50543E+14 0.25023  3.11582E-06 0.25031 ];
PU239_CAPT                (idx, [1:   4]) = [  2.72636E+16 0.03012  1.87785E-04 0.03010 ];
PU240_CAPT                (idx, [1:   4]) = [  5.74487E+15 0.06669  3.95637E-05 0.06668 ];
PU241_CAPT                (idx, [1:   4]) = [  4.74105E+14 0.22892  3.27260E-06 0.22892 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84343E+17 0.01214  1.26997E-03 0.01212 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001860 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12007E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001860 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5481591 5.48678E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3538618 3.54198E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981651 9.82441E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001860 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.33413E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36140E+20 4.3E-06  2.36140E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36847E+19 1.0E-07  9.36847E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45148E+20 0.00029  1.28945E+20 0.00020  1.62036E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38833E+20 0.00018  2.22629E+20 0.00011  1.62036E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64540E+20 0.00035  2.64540E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67590E+22 0.00025  3.48105E+22 0.00022  1.94840E+21 0.00192 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59900E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64823E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19640E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.15216E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.15216E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.00294E-01 0.04203 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30744E-01 0.01153 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.95326E-04 0.02402 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.64700E+03 0.02691 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01758E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.82190E-01 0.02982 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.05396E-01 0.02982 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52059E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99867E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.92641E-01 0.00046  9.88564E-02 0.00046  3.39889E-04 0.00960 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92697E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92679E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92697E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89960E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92341E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92126E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47712E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.48008E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90125E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90807E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93934E-03 0.00567  2.97397E-04 0.02005  9.14157E-04 0.01048  7.18424E-04 0.01308  1.57701E-03 0.00888  3.59845E-04 0.01738  7.24977E-05 0.03883 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.93196E-01 0.01408  1.24768E-02 4.4E-05  3.22517E-02 0.00014  1.06613E-01 0.00057  3.00465E-01 0.00028  1.25568E+00 0.00069  6.29716E+00 0.02882 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46569E-03 0.00789  2.63024E-04 0.02988  8.09295E-04 0.01521  6.15695E-04 0.01844  1.38786E-03 0.01227  3.21809E-04 0.02539  6.80054E-05 0.06030 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.03419E-01 0.02202  1.24767E-02 3.5E-05  3.22494E-02 0.00018  1.06541E-01 0.00074  3.00404E-01 0.00042  1.25474E+00 0.00088  7.13174E+00 0.02235 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.42393E-07 0.00139  9.41947E-07 0.00139  1.07441E-06 0.01754 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.41165E-07 0.00131  8.40768E-07 0.00131  9.58712E-07 0.01747 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42037E-03 0.00971  2.51834E-04 0.03431  8.10860E-04 0.01810  6.03469E-04 0.02173  1.38910E-03 0.01532  3.03279E-04 0.03042  6.18296E-05 0.07257 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.91348E-01 0.02577  1.24759E-02 8.7E-05  3.22364E-02 0.00024  1.06653E-01 0.00109  3.00510E-01 0.00052  1.25485E+00 0.00122  7.29371E+00 0.03165 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.33009E-07 0.01968  8.32774E-07 0.01968  8.89551E-07 0.05464 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.43644E-07 0.01968  7.43433E-07 0.01968  7.94798E-07 0.05482 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93332E-03 0.04702  1.98078E-04 0.16066  6.72764E-04 0.08554  5.26931E-04 0.11213  1.18716E-03 0.06850  2.69808E-04 0.14521  7.85803E-05 0.21671 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.68992E-01 0.11024  1.24732E-02 0.00021  3.22470E-02 0.00077  1.06551E-01 0.00326  3.02420E-01 0.00217  1.26093E+00 0.00401  7.36600E+00 0.08111 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91023E-03 0.04647  1.93849E-04 0.15863  6.60372E-04 0.08501  5.12326E-04 0.11090  1.19946E-03 0.06653  2.70496E-04 0.14560  7.37290E-05 0.21999 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.52120E-01 0.10397  1.24732E-02 0.00021  3.22481E-02 0.00077  1.06545E-01 0.00321  3.02355E-01 0.00215  1.26113E+00 0.00401  7.36600E+00 0.08111 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.52726E+03 0.04241 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.34953E-07 0.00082 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.34521E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37261E-03 0.00633 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.60731E+03 0.00629 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.88998E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.62423E-05 0.00788  6.62420E-05 0.00790  8.02638E-07 0.51929 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.78929E-05 0.02697  5.79493E-05 0.02697  4.06129E-07 0.60226 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85930E-04 0.02362  1.86269E-04 0.02358  1.13032E-04 0.50455 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64803E+01 0.01132 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51776E+01 0.00029  3.98637E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 06:41:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01559E+00  1.00381E+00  9.97574E-01  9.98681E-01  9.97461E-01  9.91871E-01  9.92699E-01  1.00380E+00  9.97784E-01  1.00012E+00  1.00260E+00  1.00336E+00  1.00071E+00  1.00331E+00  9.97936E-01  1.00049E+00  9.93679E-01  9.98534E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.31730E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56827E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30097E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44457E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75774E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52325E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51348E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.58974E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.07040E+00 0.00040  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333621 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33415E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33415E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.73406E+03 ;
RUNNING_TIME              (idx, 1)        =  5.84943E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.60215E+00  1.53150E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.81655E+02  5.84685E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.89883E-01  1.95167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.35167E-02  1.35167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.95915E+00  1.39578E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.83586E+02  8.15496E+02 ];
CPU_USAGE                 (idx, 1)        = 9.80277 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99029E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.78821E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.92014E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74171E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.86427E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01261E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38920E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87291E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56833E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.32741E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17963E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.07289E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.65067E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21214E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20621E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20371E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37597E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34332E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56263E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92496E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11886E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28114E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01331E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.26629E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.13027E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.93382E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96384E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.32779E-02 -7.16056E+27  2.22335E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03329E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.58654E+18 0.00416  1.69345E-02 0.00405 ];
U233_FISS                 (idx, [1:   4]) = [  8.24520E+19 0.00056  8.80167E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.05646E+18 0.00211  6.46536E-02 0.00209 ];
U238_FISS                 (idx, [1:   4]) = [  5.32176E+13 0.70592  5.68276E-07 0.70593 ];
PU239_FISS                (idx, [1:   4]) = [  1.04172E+17 0.01544  1.11195E-03 0.01543 ];
PU240_FISS                (idx, [1:   4]) = [  4.37803E+15 0.08129  4.67634E-05 0.08141 ];
PU241_FISS                (idx, [1:   4]) = [  6.14119E+15 0.06555  6.54860E-05 0.06546 ];
TH232_CAPT                (idx, [1:   4]) = [  9.72237E+19 0.00054  6.69527E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20448E+19 0.00147  8.29461E-02 0.00141 ];
U235_CAPT                 (idx, [1:   4]) = [  2.12233E+18 0.00355  1.46155E-02 0.00354 ];
U238_CAPT                 (idx, [1:   4]) = [  5.03736E+14 0.23521  3.46271E-06 0.23513 ];
PU239_CAPT                (idx, [1:   4]) = [  4.77454E+16 0.02519  3.28701E-04 0.02512 ];
PU240_CAPT                (idx, [1:   4]) = [  1.36182E+16 0.04435  9.37449E-05 0.04433 ];
PU241_CAPT                (idx, [1:   4]) = [  1.19004E+15 0.13768  8.18696E-06 0.13768 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83023E+17 0.01216  1.26046E-03 0.01219 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002455 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11413E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002455 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486011 5.49091E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3539244 3.54224E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 977200 9.77994E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002455 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.09896E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36155E+20 4.7E-06  2.36155E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36719E+19 1.2E-07  9.36719E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45226E+20 0.00027  1.28985E+20 0.00020  1.62403E+19 0.00167 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38897E+20 0.00016  2.22657E+20 0.00012  1.62403E+19 0.00167 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64461E+20 0.00034  2.64461E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67292E+22 0.00023  3.47732E+22 0.00022  1.95600E+21 0.00162 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.58648E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64762E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19490E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.11896E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.11896E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.79646E-01 0.04702 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30765E-01 0.01149 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.96846E-04 0.02430 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.78365E+03 0.02541 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02204E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.65207E-01 0.03131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.90418E-01 0.03131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52109E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99895E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.92944E-01 0.00045  9.88884E-02 0.00044  3.40487E-04 0.00851 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92949E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92999E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92949E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89741E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90398E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90520E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50606E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50404E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.94759E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.94235E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.99127E-03 0.00523  2.90396E-04 0.02034  9.24807E-04 0.01094  7.30218E-04 0.01287  1.61011E-03 0.00850  3.55750E-04 0.01693  7.99873E-05 0.03558 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.03870E-01 0.01341  1.24764E-02 4.5E-05  3.22386E-02 0.00012  1.06678E-01 0.00052  3.00762E-01 0.00030  1.25682E+00 0.00082  6.69368E+00 0.02463 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47496E-03 0.00719  2.59283E-04 0.02799  8.07640E-04 0.01591  6.36441E-04 0.01780  1.39462E-03 0.01175  3.07910E-04 0.02705  6.90648E-05 0.05717 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.97626E-01 0.02114  1.24770E-02 8.1E-05  3.22374E-02 0.00017  1.06688E-01 0.00079  3.00713E-01 0.00043  1.25766E+00 0.00107  7.13391E+00 0.02091 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.32589E-07 0.00136  9.32073E-07 0.00136  1.08340E-06 0.02164 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32695E-07 0.00127  8.32233E-07 0.00127  9.67464E-07 0.02167 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42672E-03 0.00854  2.57116E-04 0.03193  7.96929E-04 0.01910  6.39741E-04 0.02161  1.36109E-03 0.01429  3.05580E-04 0.03121  6.62645E-05 0.06340 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.92598E-01 0.02382  1.24770E-02 9.7E-05  3.22292E-02 0.00023  1.06638E-01 0.00103  3.01051E-01 0.00054  1.26104E+00 0.00160  6.92204E+00 0.03232 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.20347E-07 0.01964  8.20204E-07 0.01964  8.44016E-07 0.05376 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.32691E-07 0.01964  7.32563E-07 0.01964  7.54149E-07 0.05386 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01165E-03 0.04259  2.58589E-04 0.12115  6.89990E-04 0.08671  5.08714E-04 0.09450  1.30767E-03 0.05717  1.93479E-04 0.15289  5.32045E-05 0.32061 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52435E-01 0.11085  1.24772E-02 0.00013  3.22215E-02 0.00076  1.06639E-01 0.00326  3.00298E-01 0.00185  1.24906E+00 0.00375  6.73847E+00 0.11129 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03380E-03 0.04242  2.57809E-04 0.11941  7.00748E-04 0.08756  5.24194E-04 0.09629  1.30341E-03 0.05648  1.93643E-04 0.14704  5.39997E-05 0.32956 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.51855E-01 0.10898  1.24772E-02 0.00013  3.22247E-02 0.00076  1.06650E-01 0.00328  3.00081E-01 0.00181  1.24903E+00 0.00374  6.77159E+00 0.11055 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.68680E+03 0.03822 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.23594E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.24664E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43341E-03 0.00582 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.71813E+03 0.00588 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.88587E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.65369E-05 0.00727  6.65496E-05 0.00727  9.75685E-07 0.45081 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.53883E-05 0.02700  5.54745E-05 0.02700  6.02571E-07 0.51331 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.87796E-04 0.02327  1.88043E-04 0.02335  1.21672E-04 0.44503 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60175E+01 0.01129 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51348E+01 0.00025  3.96879E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 07:40:48 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01328E+00  1.00194E+00  9.94914E-01  9.98829E-01  9.88844E-01  1.00011E+00  9.96252E-01  1.00424E+00  9.98731E-01  1.00633E+00  9.96345E-01  1.00295E+00  9.99804E-01  1.00039E+00  9.98437E-01  1.00427E+00  9.95977E-01  9.98363E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.35126E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56487E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29744E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44195E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76153E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50806E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49821E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57062E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.09372E+00 0.00043  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334542 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33429E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33429E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.31152E+03 ;
RUNNING_TIME              (idx, 1)        =  6.44143E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.76365E+00  1.61500E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.40640E+02  5.89856E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.07883E-01  1.80000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.70000E-02  1.34833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.13887E+01  1.42948E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.42758E+02  8.18702E+02 ];
CPU_USAGE                 (idx, 1)        = 9.79831 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99035E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.78331E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.93982E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74747E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.14588E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97584E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36737E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92948E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57640E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.81821E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75808E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.32796E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04502E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44257E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70366E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34162E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37205E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33764E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55642E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.96504E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27177E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27616E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.06449E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.42075E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10362E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.96454E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60423E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.42879E-02 -1.16814E+28  2.26856E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03089E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.57106E+18 0.00420  1.67699E-02 0.00415 ];
U233_FISS                 (idx, [1:   4]) = [  8.05294E+19 0.00054  8.59624E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.91619E+18 0.00194  7.38269E-02 0.00185 ];
U238_FISS                 (idx, [1:   4]) = [  5.36334E+13 0.70593  5.70410E-07 0.70592 ];
PU239_FISS                (idx, [1:   4]) = [  3.32758E+17 0.00922  3.55182E-03 0.00918 ];
PU240_FISS                (idx, [1:   4]) = [  2.62030E+16 0.03128  2.79703E-04 0.03128 ];
PU241_FISS                (idx, [1:   4]) = [  4.00739E+16 0.02474  4.27669E-04 0.02470 ];
TH232_CAPT                (idx, [1:   4]) = [  9.51540E+19 0.00056  6.52106E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17235E+19 0.00149  8.03443E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.45973E+18 0.00340  1.68564E-02 0.00334 ];
U238_CAPT                 (idx, [1:   4]) = [  1.38372E+15 0.14308  9.48979E-06 0.14313 ];
PU239_CAPT                (idx, [1:   4]) = [  1.55549E+17 0.01337  1.06612E-03 0.01339 ];
PU240_CAPT                (idx, [1:   4]) = [  6.82825E+16 0.01936  4.68009E-04 0.01935 ];
PU241_CAPT                (idx, [1:   4]) = [  8.52550E+15 0.05496  5.84048E-05 0.05494 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90294E+17 0.01191  1.30440E-03 0.01196 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002878 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07192E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002878 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5491758 5.49631E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3526099 3.52867E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 985021 9.85739E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002878 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.39001E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36329E+20 4.7E-06  2.36329E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36323E+19 9.7E-08  9.36323E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45933E+20 0.00030  1.29499E+20 0.00021  1.64336E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39565E+20 0.00018  2.23132E+20 0.00012  1.64336E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65485E+20 0.00034  2.65485E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68597E+22 0.00025  3.48906E+22 0.00023  1.96902E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61706E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65736E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19544E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.99429E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.99429E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.30245E-01 0.04757 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.52702E-01 0.01023 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.21265E-04 0.02163 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.16078E+03 0.02783 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01429E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.64055E-01 0.03131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.88627E-01 0.03131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52401E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99979E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.90522E-01 0.00044  9.86180E-02 0.00043  3.43054E-04 0.00865 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90317E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.90212E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90317E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87691E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85974E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.86038E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.57426E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.57299E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.04404E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.04101E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.98454E-03 0.00532  2.91017E-04 0.01976  9.17625E-04 0.01081  7.19183E-04 0.01274  1.59942E-03 0.00843  3.72344E-04 0.01780  8.49461E-05 0.03555 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.20094E-01 0.01467  1.24772E-02 6.1E-05  3.22183E-02 0.00015  1.06710E-01 0.00055  3.01321E-01 0.00029  1.25777E+00 0.00093  6.65844E+00 0.02522 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45736E-03 0.00808  2.49667E-04 0.02910  7.84710E-04 0.01553  6.31116E-04 0.01885  1.39423E-03 0.01238  3.20965E-04 0.02591  7.66749E-05 0.04759 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.25109E-01 0.02077  1.24766E-02 5.0E-05  3.22228E-02 0.00021  1.06763E-01 0.00080  3.01164E-01 0.00044  1.25753E+00 0.00143  7.27586E+00 0.02018 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.18144E-07 0.00138  9.17798E-07 0.00139  1.01854E-06 0.01605 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17575E-07 0.00129  8.17267E-07 0.00129  9.07078E-07 0.01606 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46288E-03 0.00877  2.62139E-04 0.02960  7.99242E-04 0.01736  6.25311E-04 0.02063  1.36238E-03 0.01556  3.36352E-04 0.03033  7.74564E-05 0.06095 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.32146E-01 0.02603  1.24783E-02 0.00012  3.22111E-02 0.00028  1.06611E-01 0.00094  3.01346E-01 0.00054  1.25664E+00 0.00151  7.38763E+00 0.02718 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.04217E-07 0.01968  8.04082E-07 0.01968  8.37611E-07 0.05165 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.16092E-07 0.01968  7.15972E-07 0.01968  7.45829E-07 0.05164 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.99531E-03 0.04584  2.20708E-04 0.14193  6.61509E-04 0.08167  5.51862E-04 0.10025  1.13814E-03 0.06676  3.30603E-04 0.11492  9.24904E-05 0.23554 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.06442E-01 0.11176  1.24755E-02 0.00015  3.22115E-02 0.00071  1.06346E-01 0.00289  3.02337E-01 0.00207  1.24875E+00 0.00536  7.25654E+00 0.07452 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.95919E-03 0.04506  2.13642E-04 0.14288  6.52190E-04 0.07854  5.34010E-04 0.09819  1.13965E-03 0.06663  3.32067E-04 0.11354  8.76369E-05 0.23199 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.94588E-01 0.10839  1.24755E-02 0.00015  3.22109E-02 0.00069  1.06372E-01 0.00292  3.02344E-01 0.00206  1.24889E+00 0.00536  7.24040E+00 0.07476 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.73716E+03 0.04144 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.09761E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10110E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44998E-03 0.00515 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.79219E+03 0.00509 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92601E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73135E-05 0.00628  6.72868E-05 0.00630  1.53120E-06 0.41099 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.73991E-05 0.02367  5.73371E-05 0.02374  1.62290E-06 0.67658 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.14039E-04 0.02108  2.14290E-04 0.02103  1.46004E-04 0.40529 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58227E+01 0.01080 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49821E+01 0.00028  3.92784E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 08:40:15 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00861E+00  1.00592E+00  9.97325E-01  9.99304E-01  9.91367E-01  9.93944E-01  9.92426E-01  1.00264E+00  1.00425E+00  1.00281E+00  1.00356E+00  9.96120E-01  1.00118E+00  1.00233E+00  9.99544E-01  1.00018E+00  9.92328E-01  1.00616E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.43229E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.55677E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.28254E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.42925E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77719E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50400E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49406E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61106E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.17823E+00 0.00042  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333643 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33416E+04 0.00068 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33416E+04 0.00068 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.89174E+03 ;
RUNNING_TIME              (idx, 1)        =  7.03590E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.92047E+00  1.56817E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.99871E+02  5.92307E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.26033E-01  1.81500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.70000E-02  1.34833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.27768E+01  1.38812E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.02239E+02  8.20560E+02 ];
CPU_USAGE                 (idx, 1)        = 9.79511 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99030E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77982E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.92376E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74194E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.01306E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93006E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34302E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95914E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57324E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.49202E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38454E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.31735E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13474E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69267E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23916E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57917E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34666E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31742E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52923E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00155E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59862E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24112E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04206E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.10581E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.03313E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.98473E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32052E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.08950E-01 -2.34433E+28  2.38618E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02464E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.54042E+18 0.00382  1.64497E-02 0.00377 ];
U233_FISS                 (idx, [1:   4]) = [  7.93095E+19 0.00061  8.46922E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.13649E+18 0.00202  7.62087E-02 0.00195 ];
U238_FISS                 (idx, [1:   4]) = [  7.92083E+13 0.57553  8.53520E-07 0.57554 ];
PU239_FISS                (idx, [1:   4]) = [  6.24422E+17 0.00669  6.66793E-03 0.00667 ];
PU240_FISS                (idx, [1:   4]) = [  7.52646E+16 0.01871  8.03732E-04 0.01870 ];
PU241_FISS                (idx, [1:   4]) = [  1.34074E+17 0.01417  1.43178E-03 0.01415 ];
TH232_CAPT                (idx, [1:   4]) = [  9.32680E+19 0.00055  6.37474E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15432E+19 0.00157  7.88969E-02 0.00153 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50386E+18 0.00323  1.71140E-02 0.00323 ];
U238_CAPT                 (idx, [1:   4]) = [  2.44839E+15 0.09599  1.67274E-05 0.09600 ];
PU239_CAPT                (idx, [1:   4]) = [  2.86673E+17 0.00952  1.95936E-03 0.00951 ];
PU240_CAPT                (idx, [1:   4]) = [  1.93494E+17 0.01122  1.32257E-03 0.01123 ];
PU241_CAPT                (idx, [1:   4]) = [  2.94231E+16 0.03142  2.01156E-04 0.03145 ];
SM149_CAPT                (idx, [1:   4]) = [  1.94757E+17 0.01106  1.33094E-03 0.01101 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002487 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08522E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002487 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5492336 5.49711E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3515592 3.51840E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 994559 9.95348E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002487 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.23517E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36623E+20 4.9E-06  2.36623E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35944E+19 1.0E-07  9.35944E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46342E+20 0.00027  1.29440E+20 0.00017  1.69014E+19 0.00165 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39936E+20 0.00016  2.23035E+20 0.00010  1.69014E+19 0.00165 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66158E+20 0.00035  2.66158E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.71843E+22 0.00025  3.51834E+22 0.00022  2.00089E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.64926E+19 0.00132 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66429E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19735E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.74450E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.74450E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.62974E-01 0.04239 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19072E-01 0.00956 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.90612E-04 0.01977 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.62913E+03 0.02924 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.00471E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.96670E-01 0.02832 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.17370E-01 0.02832 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52817E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00060E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.89432E-01 0.00049  9.84924E-02 0.00048  3.43119E-04 0.00873 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.89100E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.89064E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.89100E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87385E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83381E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83482E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61560E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61370E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10548E-01 0.00140 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10481E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03444E-03 0.00540  2.88780E-04 0.01974  9.37746E-04 0.01163  7.22180E-04 0.01219  1.62515E-03 0.00876  3.80929E-04 0.01718  7.96550E-05 0.03730 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.07176E-01 0.01414  1.24792E-02 9.3E-05  3.21966E-02 0.00017  1.06760E-01 0.00050  3.01560E-01 0.00031  1.25995E+00 0.00095  6.36499E+00 0.02664 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.52541E-03 0.00763  2.50240E-04 0.02817  8.02172E-04 0.01634  6.40703E-04 0.01799  1.43550E-03 0.01250  3.29602E-04 0.02499  6.71885E-05 0.05588 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02767E-01 0.01998  1.24807E-02 0.00022  3.21989E-02 0.00024  1.06774E-01 0.00076  3.01630E-01 0.00043  1.25954E+00 0.00152  6.94748E+00 0.02178 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.14411E-07 0.00146  9.13932E-07 0.00146  1.05091E-06 0.01937 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.13232E-07 0.00133  8.12806E-07 0.00133  9.34366E-07 0.01929 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48063E-03 0.00886  2.54498E-04 0.03158  7.96322E-04 0.01956  6.13169E-04 0.02198  1.42226E-03 0.01351  3.26392E-04 0.02810  6.79904E-05 0.06370 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.09435E-01 0.02378  1.24802E-02 0.00017  3.21970E-02 0.00029  1.06679E-01 0.00099  3.01666E-01 0.00055  1.25477E+00 0.00221  7.19376E+00 0.02945 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.03657E-07 0.01978  8.03349E-07 0.01978  9.13774E-07 0.06492 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.14952E-07 0.01978  7.14678E-07 0.01978  8.13108E-07 0.06501 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00493E-03 0.04565  2.15670E-04 0.14423  7.51987E-04 0.08235  5.86282E-04 0.09830  1.12916E-03 0.07235  2.75382E-04 0.12699  4.64458E-05 0.26937 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.55604E-01 0.09235  1.24886E-02 0.00088  3.22195E-02 0.00092  1.06243E-01 0.00261  3.01487E-01 0.00207  1.26568E+00 0.00403  6.39970E+00 0.11648 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03697E-03 0.04591  2.23853E-04 0.14500  7.53108E-04 0.08085  5.82939E-04 0.09767  1.15095E-03 0.07045  2.80994E-04 0.12839  4.51333E-05 0.26629 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52160E-01 0.09059  1.24887E-02 0.00088  3.22233E-02 0.00087  1.06264E-01 0.00261  3.01628E-01 0.00209  1.26589E+00 0.00404  6.39970E+00 0.11648 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77222E+03 0.04197 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.07506E-07 0.00085 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.07100E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40621E-03 0.00623 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.75410E+03 0.00629 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.06959E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.87432E-05 0.00519  6.87095E-05 0.00525  3.29969E-06 0.28183 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.78693E-05 0.01820  5.79501E-05 0.01820  2.01474E-06 0.34842 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.77166E-04 0.01936  2.76973E-04 0.01944  3.31968E-04 0.27325 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60563E+01 0.01063 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49406E+01 0.00027  3.90149E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 09:40:01 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00978E+00  1.00010E+00  9.97930E-01  1.01247E+00  9.96147E-01  9.95127E-01  9.97533E-01  1.00175E+00  9.97077E-01  9.92276E-01  1.00056E+00  9.98106E-01  1.00068E+00  1.00278E+00  9.94294E-01  1.00569E+00  9.99033E-01  9.98665E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.53333E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.54667E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.26698E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.41657E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.79097E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50988E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49987E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.67090E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.28487E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334046 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33416E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33416E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.47443E+03 ;
RUNNING_TIME              (idx, 1)        =  7.63344E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.08233E+00  1.61867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.59404E+02  5.95332E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.44933E-01  1.89000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.70000E-02  1.34833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.42269E+01  1.44997E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.61932E+02  8.21344E+02 ];
CPU_USAGE                 (idx, 1)        = 9.79169 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99033E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77612E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.91817E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73998E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.38800E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91604E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33518E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96749E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57199E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57532E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55514E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39306E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14765E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76497E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39614E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93090E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33961E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31224E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52198E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01772E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00891E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22892E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03116E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27568E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00886E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.98942E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98062E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.64908E-01 -3.54839E+28  2.50659E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01918E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.53137E+18 0.00425  1.63544E-02 0.00417 ];
U233_FISS                 (idx, [1:   4]) = [  7.92084E+19 0.00053  8.45973E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.10053E+18 0.00185  7.58350E-02 0.00175 ];
U238_FISS                 (idx, [1:   4]) = [  1.58711E+14 0.40484  1.70124E-06 0.40487 ];
PU239_FISS                (idx, [1:   4]) = [  6.58272E+17 0.00662  7.03032E-03 0.00659 ];
PU240_FISS                (idx, [1:   4]) = [  8.72417E+16 0.01734  9.31629E-04 0.01729 ];
PU241_FISS                (idx, [1:   4]) = [  1.53089E+17 0.01279  1.63485E-03 0.01274 ];
TH232_CAPT                (idx, [1:   4]) = [  9.25883E+19 0.00054  6.32979E-01 0.00037 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15492E+19 0.00153  7.89548E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48928E+18 0.00303  1.70186E-02 0.00305 ];
U238_CAPT                 (idx, [1:   4]) = [  2.80031E+15 0.10463  1.91478E-05 0.10470 ];
PU239_CAPT                (idx, [1:   4]) = [  3.06684E+17 0.00920  2.09691E-03 0.00924 ];
PU240_CAPT                (idx, [1:   4]) = [  2.23350E+17 0.01061  1.52691E-03 0.01061 ];
PU241_CAPT                (idx, [1:   4]) = [  3.27975E+16 0.02872  2.24174E-04 0.02872 ];
SM149_CAPT                (idx, [1:   4]) = [  1.96627E+17 0.01170  1.34419E-03 0.01169 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002484 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06968E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002484 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487844 5.49258E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3513093 3.51582E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1001547 1.00230E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002484 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.21655E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36676E+20 4.7E-06  2.36676E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35893E+19 1.1E-07  9.35893E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46274E+20 0.00028  1.28933E+20 0.00019  1.73414E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39863E+20 0.00017  2.22522E+20 0.00011  1.73414E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66314E+20 0.00032  2.66314E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.75196E+22 0.00024  3.55032E+22 0.00021  2.01637E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.66934E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66557E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19959E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.53810E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.53810E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.23006E-01 0.03538 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.10143E-01 0.00883 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.73619E-04 0.01803 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.28316E+03 0.03379 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99777E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 8.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.79292E-01 0.02034 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.91116E-01 0.02034 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52888E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.88886E-01 0.00046  9.84494E-02 0.00046  3.39776E-04 0.00874 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.88859E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88737E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.88859E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87890E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83648E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83446E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61129E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61425E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11116E-01 0.00143 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10983E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.02303E-03 0.00555  2.90000E-04 0.01922  9.26282E-04 0.01132  7.13372E-04 0.01244  1.64141E-03 0.00892  3.69816E-04 0.01748  8.21572E-05 0.03691 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.09776E-01 0.01463  1.24795E-02 9.1E-05  3.21983E-02 0.00017  1.06803E-01 0.00059  3.01533E-01 0.00031  1.25971E+00 0.00115  6.44920E+00 0.02596 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46907E-03 0.00762  2.54780E-04 0.02780  7.89464E-04 0.01642  6.17183E-04 0.01827  1.41798E-03 0.01258  3.16410E-04 0.02616  7.32582E-05 0.05145 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09773E-01 0.02067  1.24793E-02 0.00014  3.22003E-02 0.00023  1.06781E-01 0.00072  3.01376E-01 0.00043  1.25874E+00 0.00163  6.99575E+00 0.02137 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.25899E-07 0.00157  9.25311E-07 0.00157  1.10914E-06 0.03331 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.22950E-07 0.00145  8.22426E-07 0.00145  9.86175E-07 0.03339 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43620E-03 0.00883  2.42087E-04 0.03380  7.77993E-04 0.01954  6.06998E-04 0.02123  1.41428E-03 0.01444  3.24642E-04 0.03088  7.01999E-05 0.06553 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06829E-01 0.02413  1.24796E-02 0.00014  3.21833E-02 0.00031  1.06839E-01 0.00105  3.01485E-01 0.00055  1.26177E+00 0.00188  6.62067E+00 0.03339 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.14702E-07 0.01997  8.14355E-07 0.01997  9.05058E-07 0.07150 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.24365E-07 0.01998  7.24059E-07 0.01999  8.04212E-07 0.07141 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.05665E-03 0.04348  2.44260E-04 0.13743  7.45721E-04 0.07974  5.87137E-04 0.08462  1.21192E-03 0.06761  2.36113E-04 0.14621  3.14935E-05 0.39021 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42750E-01 0.09280  1.24779E-02 0.00013  3.21947E-02 0.00098  1.06511E-01 0.00308  3.02086E-01 0.00201  1.25884E+00 0.00561  6.67610E+00 0.11914 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06532E-03 0.04238  2.45029E-04 0.13540  7.40855E-04 0.07913  5.91081E-04 0.08557  1.21132E-03 0.06522  2.45014E-04 0.14680  3.20198E-05 0.35297 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.48924E-01 0.09659  1.24781E-02 0.00012  3.21907E-02 0.00098  1.06507E-01 0.00303  3.02153E-01 0.00201  1.25860E+00 0.00559  6.67610E+00 0.11914 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79820E+03 0.03948 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.17259E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.15289E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.45891E-03 0.00604 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77140E+03 0.00606 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.27581E-09 0.00104 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84770E-05 0.00513  6.84776E-05 0.00513  1.71651E-06 0.38022 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.94646E-05 0.01686  5.94095E-05 0.01687  2.03787E-06 0.41136 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.56770E-04 0.01714  3.57411E-04 0.01706  1.99831E-04 0.38632 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60798E+01 0.01075 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49987E+01 0.00027  3.90224E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 20:56:40 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Sat Apr 30 10:15:56 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651276600401 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01305E+00  9.99242E-01  1.00539E+00  1.00270E+00  9.97400E-01  9.92966E-01  9.98395E-01  9.99820E-01  9.98253E-01  9.99644E-01  1.00064E+00  1.00760E+00  1.00367E+00  9.95548E-01  9.93427E-01  9.98929E-01  9.97307E-01  9.96019E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.63446E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.53655E-01 1.7E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.25578E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.40810E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.80331E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52130E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51119E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.72555E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.39457E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333759 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33391E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33391E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.82490E+03 ;
RUNNING_TIME              (idx, 1)        =  7.99262E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.28978E+00  1.28978E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.25387E+00  1.71533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.95088E+02  3.56838E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.63567E-01  1.86333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.70000E-02  1.34833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.49974E+01  7.70517E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.98541E+02  7.98541E+02 ];
CPU_USAGE                 (idx, 1)        = 9.79015 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98522E+00 4.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77467E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22944.77;
MEMSIZE                   (idx, 1)        = 22651.14;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 293.63;

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

TOT_ACTIVITY              (idx, 1)        =  6.91313E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73943E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.56228E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90693E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32945E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.97160E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57204E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.58316E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60081E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39870E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14555E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78649E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44384E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03288E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33886E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31177E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52125E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01777E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13628E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22235E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02882E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.29062E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99209E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  8.00056E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64071E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.21257E-01 -4.76089E+28  2.62784E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01490E+00 0.00079 ];
TH232_FISS                (idx, [1:   4]) = [  1.53595E+18 0.00415  1.64228E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  7.91466E+19 0.00060  8.46271E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.05172E+18 0.00189  7.53998E-02 0.00180 ];
U238_FISS                 (idx, [1:   4]) = [  2.13435E+14 0.34943  2.28118E-06 0.34943 ];
PU239_FISS                (idx, [1:   4]) = [  6.62069E+17 0.00624  7.07873E-03 0.00618 ];
PU240_FISS                (idx, [1:   4]) = [  8.78074E+16 0.01835  9.39248E-04 0.01842 ];
PU241_FISS                (idx, [1:   4]) = [  1.57622E+17 0.01248  1.68526E-03 0.01247 ];
TH232_CAPT                (idx, [1:   4]) = [  9.20877E+19 0.00056  6.28725E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15396E+19 0.00166  7.87872E-02 0.00162 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49601E+18 0.00334  1.70425E-02 0.00337 ];
U238_CAPT                 (idx, [1:   4]) = [  2.85780E+15 0.09596  1.95055E-05 0.09592 ];
PU239_CAPT                (idx, [1:   4]) = [  3.07308E+17 0.00890  2.09834E-03 0.00893 ];
PU240_CAPT                (idx, [1:   4]) = [  2.29378E+17 0.01118  1.56595E-03 0.01114 ];
PU241_CAPT                (idx, [1:   4]) = [  3.36718E+16 0.02735  2.29856E-04 0.02734 ];
SM149_CAPT                (idx, [1:   4]) = [  2.03532E+17 0.01163  1.38960E-03 0.01161 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001725 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07030E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001725 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487076 5.49215E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3503813 3.50691E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1010836 1.01164E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001725 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -6.70552E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 2.4E-09  3.61697E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36684E+20 4.8E-06  2.36684E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35889E+19 1.0E-07  9.35889E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46493E+20 0.00029  1.28526E+20 0.00018  1.79675E+19 0.00159 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40082E+20 0.00018  2.22114E+20 0.00011  1.79675E+19 0.00159 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66685E+20 0.00033  2.66685E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.78822E+22 0.00023  3.58221E+22 0.00021  2.06002E+21 0.00158 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.69797E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67062E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20429E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.36408E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.36408E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.07425E-01 0.03191 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06965E-01 0.00785 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.71989E-04 0.01538 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.45817E+03 0.03656 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.98844E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.20913E-01 0.01546 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.27695E-01 0.01546 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52897E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.86899E-01 0.00046  9.82010E-02 0.00046  3.44285E-04 0.00918 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.87214E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.87531E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.87214E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87033E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83709E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83829E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61030E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.60811E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11425E-01 0.00132 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11220E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.04178E-03 0.00588  2.90549E-04 0.02025  9.29872E-04 0.01142  7.25592E-04 0.01298  1.63950E-03 0.00887  3.72344E-04 0.01865  8.39313E-05 0.03585 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10201E-01 0.01486  1.24813E-02 0.00012  3.21940E-02 0.00016  1.06902E-01 0.00057  3.01343E-01 0.00030  1.25677E+00 0.00108  6.31394E+00 0.02525 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48237E-03 0.00833  2.54142E-04 0.02786  8.00505E-04 0.01759  6.09819E-04 0.01820  1.42929E-03 0.01292  3.12085E-04 0.02519  7.65270E-05 0.05233 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.19171E-01 0.02232  1.24818E-02 0.00017  3.21921E-02 0.00023  1.06892E-01 0.00081  3.01253E-01 0.00044  1.25837E+00 0.00138  6.74751E+00 0.02195 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.40496E-07 0.00165  9.40097E-07 0.00166  1.05602E-06 0.01686 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.34055E-07 0.00155  8.33702E-07 0.00155  9.36487E-07 0.01682 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48886E-03 0.00924  2.56133E-04 0.03427  8.05317E-04 0.01921  6.40214E-04 0.02225  1.39455E-03 0.01439  3.16129E-04 0.03168  7.65214E-05 0.06029 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.16771E-01 0.02430  1.24818E-02 0.00018  3.21967E-02 0.00028  1.06994E-01 0.00100  3.01292E-01 0.00050  1.25800E+00 0.00192  7.04704E+00 0.02835 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.22378E-07 0.02014  8.22269E-07 0.02014  8.47929E-07 0.05665 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.29484E-07 0.02016  7.29389E-07 0.02016  7.51681E-07 0.05661 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16921E-03 0.04575  2.38641E-04 0.13401  7.02317E-04 0.08226  5.45583E-04 0.09044  1.33323E-03 0.06412  2.85821E-04 0.11898  6.36156E-05 0.25602 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.10460E-01 0.10929  1.24884E-02 0.00078  3.21832E-02 0.00094  1.06539E-01 0.00290  3.00282E-01 0.00202  1.26365E+00 0.00391  6.14859E+00 0.09698 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.17274E-03 0.04502  2.39622E-04 0.13204  7.10424E-04 0.08182  5.40959E-04 0.08843  1.32435E-03 0.06319  2.89930E-04 0.11893  6.74574E-05 0.23606 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.24882E-01 0.11011  1.24885E-02 0.00078  3.21833E-02 0.00094  1.06553E-01 0.00289  3.00310E-01 0.00201  1.26366E+00 0.00391  6.12463E+00 0.09684 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.90210E+03 0.04241 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.28022E-07 0.00088 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.23008E-07 0.00074 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53908E-03 0.00681 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81491E+03 0.00697 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.50347E-09 0.00113 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.93481E-05 0.00428  6.93218E-05 0.00430  4.11951E-06 0.24017 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.27402E-05 0.01565  6.27485E-05 0.01584  4.33191E-06 0.29863 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.50938E-04 0.01503  4.51052E-04 0.01514  4.29044E-04 0.23700 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59923E+01 0.01147 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51119E+01 0.00028  3.90355E+01 0.00038 ];

