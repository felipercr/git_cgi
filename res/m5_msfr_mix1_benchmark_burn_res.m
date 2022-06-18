
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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 03:09:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04087E+00  1.00750E+00  9.99840E-01  1.00432E+00  9.97309E-01  9.96884E-01  1.00747E+00  9.97113E-01  1.00256E+00  1.00041E+00  9.85168E-01  9.83723E-01  9.91936E-01  1.00155E+00  1.00262E+00  9.86335E-01  9.92639E-01  1.00176E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.80253E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71975E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40357E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54200E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60245E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46017E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45112E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.11564E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.96187E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000248 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00023E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00023E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.73897E+02 ;
RUNNING_TIME              (idx, 1)        =  1.84722E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.10583E-01  1.10583E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.72826E+01  1.72826E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.17250E-01  5.78667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.84680E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.41401 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97569E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.37934E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

NORM_COEF                 (idx, [1:   4]) = [  1.17492E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.54842E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.51640E+18 0.00397  1.61721E-02 0.00397 ];
U233_FISS                 (idx, [1:   4]) = [  9.22564E+19 0.00050  9.83828E-01 6.5E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.03578E+19 0.00053  7.51878E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34458E+19 0.00130  1.11887E-01 0.00126 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001166 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15240E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001166 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5108716 5.11427E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3986661 3.99064E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905789 9.06617E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001166 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.86265E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36452E+20 3.7E-06  2.36452E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.3E-07  9.37764E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.20197E+20 0.00031  1.06876E+20 0.00023  1.33218E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.13974E+20 0.00018  2.00652E+20 0.00012  1.33218E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.34984E+20 0.00036  2.34984E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.09991E+22 0.00026  2.93726E+22 0.00023  1.62644E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.13052E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.35279E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.04709E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51104E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.21673E-01 0.08263 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.78991E-01 0.00811 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.87739E-04 0.01494 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.28386E+03 0.01856 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09348E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.14280E-01 0.07108 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.85806E-01 0.07108 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52145E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.00613E+00 0.00040  4.01194E-02 0.00040  1.30031E-04 0.00873 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.00616E+00 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  1.00631E+00 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.00616E+00 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.10647E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03237E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03341E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32481E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32319E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47793E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.47493E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.20810E-03 0.00555  2.67864E-04 0.01990  7.86951E-04 0.01152  5.86138E-04 0.01305  1.25612E-03 0.00904  2.64057E-04 0.01937  4.69767E-05 0.04504 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40195E-01 0.01435  1.24513E-02 0.00200  3.23322E-02 9.3E-05  1.06163E-01 0.00063  2.97142E-01 0.00024  1.22793E+00 0.00349  3.96850E+00 0.04431 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19144E-03 0.00825  2.73908E-04 0.02958  8.01259E-04 0.01732  5.79962E-04 0.01961  1.23897E-03 0.01373  2.51670E-04 0.02792  4.56705E-05 0.06821 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.31049E-01 0.02033  1.24761E-02 3.3E-05  3.23298E-02 0.00013  1.06160E-01 0.00082  2.97119E-01 0.00036  1.23514E+00 0.00030  6.30008E+00 0.02877 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.21096E-07 0.00120  8.20838E-07 0.00120  9.03634E-07 0.01587 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.26063E-07 0.00112  8.25804E-07 0.00113  9.08968E-07 0.01585 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22936E-03 0.00897  2.71099E-04 0.03070  7.88111E-04 0.01819  5.98616E-04 0.02113  1.25087E-03 0.01460  2.73735E-04 0.02928  4.69289E-05 0.07235 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.40196E-01 0.02383  1.24760E-02 4.4E-05  3.23304E-02 0.00016  1.06014E-01 0.00094  2.97024E-01 0.00041  1.23587E+00 0.00037  6.22156E+00 0.04189 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.27914E-07 0.01532  7.27712E-07 0.01532  7.42800E-07 0.04980 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.32442E-07 0.01531  7.32238E-07 0.01531  7.47352E-07 0.04981 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.22870E-03 0.04029  3.25865E-04 0.11559  7.76859E-04 0.07776  6.08250E-04 0.08578  1.19020E-03 0.06192  2.86617E-04 0.12839  4.09081E-05 0.35970 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.19642E-01 0.08306  1.24772E-02 9.0E-05  3.23367E-02 0.00051  1.05921E-01 0.00278  2.97186E-01 0.00142  1.23518E+00 0.00102  5.02331E+00 0.15431 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.21905E-03 0.03943  3.10698E-04 0.11237  7.65883E-04 0.07742  6.07861E-04 0.08663  1.20897E-03 0.06022  2.84247E-04 0.12573  4.13898E-05 0.35531 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.19679E-01 0.08345  1.24771E-02 9.2E-05  3.23379E-02 0.00051  1.05892E-01 0.00275  2.97162E-01 0.00141  1.23515E+00 0.00102  5.02331E+00 0.15431 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.45939E+03 0.03759 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.14643E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.19581E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28423E-03 0.00559 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03216E+03 0.00560 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.20749E-09 0.00131 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.25550E-05 0.00450  7.25464E-05 0.00452  2.89672E-06 0.22968 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.69660E-05 0.01564  7.68758E-05 0.01578  3.28694E-06 0.34897 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.69521E-04 0.01470  4.68931E-04 0.01468  6.55324E-04 0.23757 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72026E+01 0.01159 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45112E+01 0.00027  4.02902E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 03:37:00 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01465E+00  9.92920E-01  9.92503E-01  1.00619E+00  9.97026E-01  1.00275E+00  9.88324E-01  1.00036E+00  1.00250E+00  1.00055E+00  1.00172E+00  1.00101E+00  1.00459E+00  1.00395E+00  1.00195E+00  1.00005E+00  9.91581E-01  9.97385E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.79903E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72010E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40411E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54237E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60378E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46639E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45730E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.12558E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.96068E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000544 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00042E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00042E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.47598E+02 ;
RUNNING_TIME              (idx, 1)        =  4.59963E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.02017E-01  9.14333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.46903E+01  2.74077E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.69833E-02  1.69833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.77700E-01  6.03500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.59923E+01  2.72076E+02 ];
CPU_USAGE                 (idx, 1)        = 9.73118 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98244E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.72626E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  5.74232E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60132E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.64478E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.44707E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.01916E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26485E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46892E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.42050E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.64646E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.35726E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.68065E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06324E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17839E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44274E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35720E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17597E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55495E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59954E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38178E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.37595E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.50882E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50889E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.23648E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.18776E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08018E-04 -2.38500E+25  2.20820E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.66765E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.53848E+18 0.00378  1.64009E-02 0.00372 ];
U233_FISS                 (idx, [1:   4]) = [  9.22348E+19 0.00051  9.83311E-01 6.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.55258E+14 0.28644  3.79372E-06 0.28687 ];
TH232_CAPT                (idx, [1:   4]) = [  9.15127E+19 0.00054  7.47625E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34158E+19 0.00127  1.09603E-01 0.00119 ];
U235_CAPT                 (idx, [1:   4]) = [  1.41726E+14 0.40623  1.15409E-06 0.40629 ];
SM149_CAPT                (idx, [1:   4]) = [  1.24231E+16 0.04419  1.01529E-04 0.04428 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002110 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15980E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002110 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5147788 5.15282E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3945020 3.94869E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909302 9.10095E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002110 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.36556E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.3E-09  3.52483E-02 2.3E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36446E+20 3.8E-06  2.36446E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37761E+19 1.2E-07  9.37761E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.22377E+20 0.00031  1.08877E+20 0.00022  1.35000E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.16153E+20 0.00017  2.02653E+20 0.00012  1.35000E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.37551E+20 0.00036  2.37551E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.13627E+22 0.00027  2.97167E+22 0.00024  1.64594E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.16205E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.37773E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05998E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51194E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51194E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.66461E-01 0.07514 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.82825E-01 0.00851 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.48944E-04 0.01557 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.99645E+03 0.01984 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09001E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99988E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.72291E-01 0.06237 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.38407E-01 0.06238 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52139E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.95675E-01 0.00044  3.96981E-02 0.00043  1.27532E-04 0.00832 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.95585E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.95414E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.95585E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09529E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03244E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03339E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32471E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32322E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48730E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48192E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.24884E-03 0.00530  2.62555E-04 0.01931  7.99754E-04 0.01144  6.00365E-04 0.01272  1.26734E-03 0.00877  2.73320E-04 0.01939  4.55079E-05 0.04714 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.36942E-01 0.01425  1.24253E-02 0.00284  3.23368E-02 9.4E-05  1.06055E-01 0.00060  2.97440E-01 0.00027  1.23323E+00 0.00202  3.77448E+00 0.04687 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25637E-03 0.00772  2.58337E-04 0.02975  8.02513E-04 0.01663  6.00615E-04 0.01935  1.26782E-03 0.01295  2.76351E-04 0.02935  5.07352E-05 0.06541 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48291E-01 0.02175  1.24749E-02 4.0E-05  3.23338E-02 0.00012  1.06070E-01 0.00089  2.97392E-01 0.00039  1.23561E+00 0.00029  6.39779E+00 0.02911 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.28722E-07 0.00124  8.28436E-07 0.00125  9.24005E-07 0.01726 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.25054E-07 0.00115  8.24769E-07 0.00116  9.19870E-07 0.01724 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20281E-03 0.00849  2.63075E-04 0.03275  8.01767E-04 0.01734  5.79563E-04 0.02136  1.24593E-03 0.01426  2.64017E-04 0.03051  4.84623E-05 0.07114 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44515E-01 0.02394  1.24754E-02 4.8E-05  3.23389E-02 0.00015  1.06076E-01 0.00101  2.97442E-01 0.00046  1.23559E+00 0.00039  6.49405E+00 0.03966 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.35283E-07 0.01531  7.34992E-07 0.01531  7.84147E-07 0.05152 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.32428E-07 0.01529  7.32137E-07 0.01530  7.81674E-07 0.05160 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85579E-03 0.04247  2.55169E-04 0.14492  7.07539E-04 0.07790  4.28860E-04 0.09821  1.18615E-03 0.06427  2.32645E-04 0.14736  4.54259E-05 0.30545 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.58121E-01 0.11039  1.24750E-02 0.00015  3.23371E-02 0.00049  1.05983E-01 0.00313  2.97914E-01 0.00162  1.23590E+00 0.00110  6.14486E+00 0.13882 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.84091E-03 0.04205  2.54681E-04 0.14081  7.16448E-04 0.07754  4.21065E-04 0.09838  1.17450E-03 0.06355  2.33081E-04 0.14166  4.11306E-05 0.30305 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.55880E-01 0.10591  1.24750E-02 0.00015  3.23379E-02 0.00050  1.05997E-01 0.00315  2.97910E-01 0.00160  1.23582E+00 0.00110  6.14486E+00 0.13882 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.91297E+03 0.04038 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.24163E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.20517E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22834E-03 0.00565 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.91766E+03 0.00566 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16548E-09 0.00117 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.13406E-05 0.00448  7.13467E-05 0.00448  1.69406E-06 0.29377 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.53027E-05 0.01558  7.53921E-05 0.01559  7.64414E-07 0.46929 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.32850E-04 0.01508  4.32957E-04 0.01514  4.13049E-04 0.29986 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73301E+01 0.01118 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45730E+01 0.00027  4.03133E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 04:06:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01121E+00  1.01104E+00  9.90875E-01  1.00258E+00  1.00010E+00  1.00452E+00  1.00019E+00  9.92957E-01  1.00196E+00  1.00188E+00  9.95864E-01  9.96722E-01  9.99147E-01  9.96738E-01  9.94933E-01  1.00024E+00  1.00321E+00  9.95832E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.78585E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72142E-01 9.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40408E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54173E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60060E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47859E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46950E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.15010E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.93799E-01 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000128 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00034E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00034E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.37172E+02 ;
RUNNING_TIME              (idx, 1)        =  7.51097E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.96717E-01  9.47000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.36830E+01  2.89927E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.34333E-02  1.64500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.40317E-01  6.25333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.51057E+01  4.05332E+02 ];
CPU_USAGE                 (idx, 1)        = 9.81460 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98326E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.81716E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.35088E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65027E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.21635E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.82620E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.27613E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49398E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49188E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.69771E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.38401E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.80226E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.84481E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89523E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.59953E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49318E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34334E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33178E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54032E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.69814E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17747E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.98173E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.50359E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.31625E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.43803E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.20287E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45045E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.77072E-04 -1.27416E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.83770E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.55712E+18 0.00411  1.65959E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  9.21342E+19 0.00051  9.82089E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  9.48528E+15 0.05294  1.01050E-04 0.05292 ];
TH232_CAPT                (idx, [1:   4]) = [  9.29625E+19 0.00054  7.42886E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34055E+19 0.00132  1.07132E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  3.60626E+15 0.08359  2.88504E-05 0.08358 ];
SM149_CAPT                (idx, [1:   4]) = [  5.76188E+16 0.02107  4.60520E-04 0.02110 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001691 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15768E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001691 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5196291 5.20162E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3895833 3.89964E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909567 9.10307E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001691 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.78814E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.1E-09  3.52483E-02 2.1E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36436E+20 3.8E-06  2.36436E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37756E+19 1.3E-07  9.37756E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25146E+20 0.00031  1.11499E+20 0.00023  1.36472E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.18921E+20 0.00018  2.05274E+20 0.00012  1.36472E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40574E+20 0.00037  2.40574E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.18135E+22 0.00026  3.01506E+22 0.00024  1.66287E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.19008E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.40822E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07642E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51433E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51433E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.88884E-01 0.05063 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.18498E-01 0.01058 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.86765E-04 0.02008 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.33430E+03 0.01961 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08974E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.14664E-01 0.03904 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.58747E-01 0.03904 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52129E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99674E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.83107E-01 0.00042  3.92000E-02 0.00040  1.27975E-04 0.00840 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.82941E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.82862E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.82941E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08139E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04221E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04152E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31189E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31249E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48790E-01 0.00166 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49042E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.31273E-03 0.00532  2.68977E-04 0.01888  7.97275E-04 0.01077  6.15709E-04 0.01264  1.30953E-03 0.00916  2.65807E-04 0.01972  5.54335E-05 0.04443 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.51266E-01 0.01507  1.23750E-02 0.00402  3.23302E-02 8.9E-05  1.06072E-01 0.00058  2.97484E-01 0.00027  1.22810E+00 0.00349  4.16367E+00 0.04156 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26777E-03 0.00762  2.65931E-04 0.02815  7.76132E-04 0.01590  6.17397E-04 0.01862  1.28677E-03 0.01289  2.65224E-04 0.02878  5.63108E-05 0.06551 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.55250E-01 0.02341  1.24751E-02 3.9E-05  3.23311E-02 0.00014  1.06040E-01 0.00077  2.97366E-01 0.00037  1.23546E+00 0.00031  6.47446E+00 0.02667 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.44834E-07 0.00127  8.44539E-07 0.00128  9.41226E-07 0.01685 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.30475E-07 0.00117  8.30184E-07 0.00117  9.25481E-07 0.01690 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26226E-03 0.00858  2.60993E-04 0.03029  7.78284E-04 0.01758  6.14917E-04 0.01998  1.29631E-03 0.01370  2.61088E-04 0.02960  5.06689E-05 0.07333 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44598E-01 0.02447  1.24744E-02 5.6E-05  3.23282E-02 0.00016  1.05896E-01 0.00091  2.97274E-01 0.00043  1.23554E+00 0.00038  6.16847E+00 0.04082 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.42925E-07 0.01536  7.42729E-07 0.01536  7.65497E-07 0.05201 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.30486E-07 0.01535  7.30292E-07 0.01535  7.53439E-07 0.05207 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.81419E-03 0.04082  1.92762E-04 0.15704  6.87798E-04 0.08076  5.82661E-04 0.08563  1.09544E-03 0.05883  2.02437E-04 0.13190  5.30950E-05 0.32595 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.21053E-01 0.07640  1.24763E-02 0.00013  3.23271E-02 0.00046  1.06304E-01 0.00319  2.97347E-01 0.00144  1.23605E+00 0.00108  6.09731E+00 0.12169 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.79873E-03 0.04036  1.90894E-04 0.15233  6.81078E-04 0.07793  5.74718E-04 0.08399  1.09463E-03 0.05806  2.02874E-04 0.13270  5.45316E-05 0.31023 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.25758E-01 0.08264  1.24763E-02 0.00013  3.23281E-02 0.00046  1.06301E-01 0.00320  2.97354E-01 0.00143  1.23594E+00 0.00109  6.12932E+00 0.12140 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81545E+03 0.03808 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.38892E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.24653E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22732E-03 0.00587 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84883E+03 0.00598 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99186E-09 0.00099 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.86480E-05 0.00754  6.86569E-05 0.00753  9.52743E-07 0.38757 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.95135E-05 0.02296  6.95057E-05 0.02297  4.88826E-07 0.68273 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.79162E-04 0.01927  2.79390E-04 0.01934  2.22507E-04 0.37758 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72344E+01 0.01165 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46950E+01 0.00027  4.05049E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 04:35:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01095E+00  1.00550E+00  9.94652E-01  9.99910E-01  1.00322E+00  9.99869E-01  9.97338E-01  1.00505E+00  1.00915E+00  9.98391E-01  9.99085E-01  9.91369E-01  1.00364E+00  9.91957E-01  9.96791E-01  9.90667E-01  9.93933E-01  1.00854E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.77571E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72243E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40367E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54091E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59839E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49220E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48314E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17793E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.91286E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000136 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00033E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00033E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.03141E+03 ;
RUNNING_TIME              (idx, 1)        =  1.04693E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.92700E-01  9.59833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.03143E+02  2.94595E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.77333E-02  1.43000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.00717E-01  6.03167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.04689E+02  4.24821E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85177 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98344E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.85773E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.46699E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66565E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.08747E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.86429E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.30184E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57159E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50429E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.75016E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.54478E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.95731E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.04071E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79177E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74069E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48770E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.35835E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34328E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55578E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39873E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42532E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.05386E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53084E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.10217E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.59865E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.21258E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.13156E-03 -2.49844E+26  2.21046E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.98204E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.57452E+18 0.00398  1.67979E-02 0.00392 ];
U233_FISS                 (idx, [1:   4]) = [  9.19014E+19 0.00053  9.80504E-01 7.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.09556E+16 0.02344  4.36996E-04 0.02343 ];
TH232_CAPT                (idx, [1:   4]) = [  9.39930E+19 0.00053  7.39639E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34015E+19 0.00132  1.05457E-01 0.00125 ];
U235_CAPT                 (idx, [1:   4]) = [  1.40258E+16 0.04057  1.10498E-04 0.04064 ];
SM149_CAPT                (idx, [1:   4]) = [  9.69348E+16 0.01541  7.62889E-04 0.01541 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001648 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14359E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001648 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5234756 5.24008E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3861189 3.86487E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905703 9.06485E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001648 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.72529E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36426E+20 3.6E-06  2.36426E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37749E+19 1.1E-07  9.37749E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.27097E+20 0.00030  1.13293E+20 0.00021  1.38032E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20871E+20 0.00017  2.07068E+20 0.00012  1.38032E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42517E+20 0.00036  2.42517E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.21284E+22 0.00025  3.04503E+22 0.00023  1.67809E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.19846E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42856E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08841E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51443E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51443E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.96213E-01 0.05240 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24187E-01 0.01155 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.25704E-04 0.02101 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.24221E+03 0.02022 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09355E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.85143E-01 0.04082 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.32078E-01 0.04082 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52121E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.74539E-01 0.00043  3.88512E-02 0.00043  1.25485E-04 0.00889 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.74662E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.74947E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.74662E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07179E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04876E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04763E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30329E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30448E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50099E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49646E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34231E-03 0.00534  2.74081E-04 0.01964  8.19660E-04 0.01146  6.06478E-04 0.01241  1.30566E-03 0.00863  2.81297E-04 0.01953  5.51345E-05 0.04481 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53309E-01 0.01625  1.24753E-02 2.9E-05  3.23375E-02 9.7E-05  1.06261E-01 0.00066  2.97412E-01 0.00027  1.23124E+00 0.00285  4.02604E+00 0.04335 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24606E-03 0.00838  2.65732E-04 0.02824  7.90578E-04 0.01699  5.84864E-04 0.01992  1.27364E-03 0.01258  2.73144E-04 0.02753  5.81043E-05 0.06180 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.61486E-01 0.02276  1.24751E-02 3.9E-05  3.23293E-02 0.00012  1.06280E-01 0.00090  2.97414E-01 0.00037  1.23631E+00 0.00032  6.34248E+00 0.02786 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.55858E-07 0.00130  8.55553E-07 0.00130  9.40961E-07 0.01596 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.33982E-07 0.00121  8.33685E-07 0.00121  9.16972E-07 0.01593 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22369E-03 0.00877  2.57378E-04 0.03224  7.93573E-04 0.01814  5.99153E-04 0.02046  1.25192E-03 0.01451  2.65293E-04 0.03251  5.63725E-05 0.06933 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.54677E-01 0.02565  1.24748E-02 5.3E-05  3.23374E-02 0.00017  1.06158E-01 0.00110  2.97354E-01 0.00047  1.23561E+00 0.00048  6.44705E+00 0.03863 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.58790E-07 0.01534  7.58631E-07 0.01534  7.58762E-07 0.05348 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.39594E-07 0.01533  7.39441E-07 0.01533  7.39698E-07 0.05353 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.96975E-03 0.03888  2.49774E-04 0.11753  7.11973E-04 0.08206  4.62891E-04 0.09287  1.18978E-03 0.06189  2.74608E-04 0.11556  8.07275E-05 0.22871 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.72799E-01 0.11879  1.24729E-02 0.00018  3.23445E-02 0.00056  1.06323E-01 0.00340  2.97981E-01 0.00168  1.23456E+00 0.00108  7.09209E+00 0.08882 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98069E-03 0.03849  2.55287E-04 0.11746  7.02936E-04 0.07896  4.68732E-04 0.09452  1.19273E-03 0.06129  2.81138E-04 0.11460  7.98691E-05 0.22452 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.69599E-01 0.11897  1.24727E-02 0.00018  3.23439E-02 0.00056  1.06296E-01 0.00337  2.97915E-01 0.00165  1.23456E+00 0.00108  7.09209E+00 0.08882 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.93640E+03 0.03616 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.48892E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.27200E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25640E-03 0.00607 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.83673E+03 0.00611 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92809E-09 0.00087 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77783E-05 0.00830  6.77960E-05 0.00829  1.12036E-06 0.36461 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.38526E-05 0.02580  6.39597E-05 0.02583  5.75620E-07 0.46675 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.16535E-04 0.02065  2.16460E-04 0.02074  2.48917E-04 0.35421 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75991E+01 0.01200 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48314E+01 0.00026  4.06490E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 05:06:13 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01615E+00  9.96324E-01  9.89962E-01  1.00579E+00  1.00325E+00  9.97615E-01  9.96978E-01  9.97990E-01  9.97157E-01  1.01009E+00  1.00803E+00  9.96381E-01  1.00238E+00  1.00094E+00  9.92461E-01  9.97558E-01  9.85225E-01  1.00573E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76853E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72315E-01 9.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40271E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53969E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59376E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50638E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49726E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20807E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.91180E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000365 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00033E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00033E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.33483E+03 ;
RUNNING_TIME              (idx, 1)        =  1.35210E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.95633E-01  1.02933E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.33518E+02  3.03758E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.65333E-02  1.88000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.67633E-01  6.68333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.35205E+02  4.31018E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87222 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98372E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88024E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.49180E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65920E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.47555E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88825E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31700E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57221E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49612E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.92599E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.66383E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.22848E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.14832E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69191E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.84889E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93480E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.31889E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31292E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51413E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35091E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81303E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.04963E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.49939E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.48414E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.64514E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.23038E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.30439E-03 -5.08801E+26  2.21305E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.22432E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.58406E+18 0.00404  1.68950E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  9.15919E+19 0.00050  9.77020E-01 8.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.56276E+17 0.01283  1.66722E-03 0.01284 ];
TH232_CAPT                (idx, [1:   4]) = [  9.57573E+19 0.00052  7.35717E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33741E+19 0.00138  1.02758E-01 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  5.31908E+16 0.02103  4.08630E-04 0.02101 ];
SM149_CAPT                (idx, [1:   4]) = [  1.34961E+17 0.01332  1.03696E-03 0.01333 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001643 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16185E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001643 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5283777 5.28927E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3806047 3.80970E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911819 9.12649E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001643 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.01166E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36411E+20 3.7E-06  2.36411E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37728E+19 1.1E-07  9.37728E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.30218E+20 0.00029  1.16167E+20 0.00022  1.40509E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23991E+20 0.00017  2.09940E+20 0.00012  1.40509E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46076E+20 0.00035  2.46076E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.26662E+22 0.00025  3.09675E+22 0.00023  1.69860E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24593E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46450E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10788E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51195E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51195E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.68469E-01 0.04883 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27316E-01 0.01265 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.02224E-04 0.02366 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.21667E+03 0.01936 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08738E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.96050E-01 0.03936 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.41774E-01 0.03936 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52111E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99680E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.60520E-01 0.00044  3.82957E-02 0.00043  1.23239E-04 0.00922 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.60390E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.60782E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.60390E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05680E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04957E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05124E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30228E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29979E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51888E-01 0.00164 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51746E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.37910E-03 0.00569  2.71156E-04 0.01991  8.35054E-04 0.01126  6.12667E-04 0.01324  1.33041E-03 0.00904  2.79417E-04 0.01945  5.03980E-05 0.04597 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40115E-01 0.01493  1.23757E-02 0.00402  3.23294E-02 9.1E-05  1.06133E-01 0.00061  2.97793E-01 0.00028  1.23437E+00 0.00202  3.79401E+00 0.04480 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23176E-03 0.00830  2.58830E-04 0.02956  7.82075E-04 0.01600  5.97877E-04 0.01989  1.28074E-03 0.01295  2.59969E-04 0.02945  5.22702E-05 0.06901 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44225E-01 0.02185  1.24752E-02 4.1E-05  3.23319E-02 0.00013  1.06188E-01 0.00086  2.97857E-01 0.00041  1.23692E+00 0.00033  6.11268E+00 0.02903 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68656E-07 0.00117  8.68408E-07 0.00117  9.46664E-07 0.01643 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.34286E-07 0.00109  8.34048E-07 0.00109  9.09226E-07 0.01643 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19864E-03 0.00934  2.50823E-04 0.03162  7.63213E-04 0.01866  5.85313E-04 0.02311  1.28497E-03 0.01449  2.69225E-04 0.03132  4.50979E-05 0.07334 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.35573E-01 0.02206  1.24749E-02 5.7E-05  3.23231E-02 0.00017  1.06267E-01 0.00118  2.97674E-01 0.00046  1.23721E+00 0.00051  6.04990E+00 0.04370 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.70572E-07 0.01544  7.70309E-07 0.01544  7.74442E-07 0.05170 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.40296E-07 0.01543  7.40043E-07 0.01543  7.44434E-07 0.05178 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.73485E-03 0.04234  2.14908E-04 0.14696  6.28757E-04 0.08261  5.88072E-04 0.08650  1.04398E-03 0.06478  2.22153E-04 0.13252  3.69827E-05 0.33064 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.61099E-01 0.10222  1.24741E-02 0.00018  3.23070E-02 0.00044  1.05975E-01 0.00284  2.98385E-01 0.00168  1.23893E+00 0.00113  5.49603E+00 0.12822 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.74992E-03 0.04122  2.12644E-04 0.14518  6.35022E-04 0.08047  5.92275E-04 0.08544  1.03933E-03 0.06284  2.32990E-04 0.12779  3.76654E-05 0.31478 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.64635E-01 0.10172  1.24742E-02 0.00018  3.23069E-02 0.00043  1.05970E-01 0.00283  2.98370E-01 0.00167  1.23882E+00 0.00113  5.49603E+00 0.12822 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.58254E+03 0.04018 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.62588E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.28448E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20298E-03 0.00624 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.71405E+03 0.00628 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91601E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63340E-05 0.00938  6.63340E-05 0.00938  0.00000E+00 0.0E+00 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.93798E-05 0.02873  5.93798E-05 0.02873  0.00000E+00 0.0E+00 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.95391E-04 0.02258  1.96054E-04 0.02258  0.00000E+00 0.0E+00 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72545E+01 0.01191 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49726E+01 0.00026  4.07375E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 05:37:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01412E+00  9.95550E-01  9.91605E-01  1.00159E+00  9.95313E-01  1.00390E+00  9.89824E-01  1.00841E+00  1.00163E+00  1.00145E+00  9.97608E-01  1.00166E+00  1.00687E+00  9.96080E-01  9.94904E-01  1.00651E+00  9.93932E-01  9.99037E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76723E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72328E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39908E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53608E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59584E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51821E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50909E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24264E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.93786E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000163 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00030E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00030E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.64198E+03 ;
RUNNING_TIME              (idx, 1)        =  1.66094E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.91667E-01  9.60333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.64267E+02  3.07488E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  8.34667E-02  1.69333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.29483E-01  6.17333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.66090E+02  4.40489E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88583 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98298E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89488E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.56351E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66926E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.03936E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92623E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34224E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.60551E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50324E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.08990E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.77574E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.63699E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.32134E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.24752E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94332E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34887E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32986E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32095E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52508E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02237E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17042E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09179E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53276E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.28174E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.75318E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.24535E+16 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85969E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.45922E-03 -7.63783E+26  2.21560E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.44293E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.60234E+18 0.00411  1.70810E-02 0.00405 ];
U233_FISS                 (idx, [1:   4]) = [  9.12784E+19 0.00053  9.73093E-01 9.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.28776E+17 0.00886  3.50444E-03 0.00880 ];
TH232_CAPT                (idx, [1:   4]) = [  9.73244E+19 0.00055  7.32876E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33170E+19 0.00139  1.00282E-01 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  1.18780E+17 0.01440  8.94316E-04 0.01437 ];
SM149_CAPT                (idx, [1:   4]) = [  1.54905E+17 0.01281  1.16678E-03 0.01283 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001495 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13769E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001495 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5326350 5.33178E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3762511 3.76614E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912634 9.13458E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001495 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.14204E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36397E+20 3.7E-06  2.36397E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37700E+19 1.1E-07  9.37700E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32787E+20 0.00028  1.18586E+20 0.00021  1.42010E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26557E+20 0.00017  2.12356E+20 0.00012  1.42010E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49070E+20 0.00038  2.49070E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.31445E+22 0.00026  3.14278E+22 0.00024  1.71673E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.27528E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.49310E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12428E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50821E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50821E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.31902E-01 0.05157 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.42968E-01 0.01121 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94889E-04 0.02390 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.43031E+03 0.01757 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08657E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.57970E-01 0.04182 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.06990E-01 0.04182 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52103E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.49355E-01 0.00042  3.78552E-02 0.00042  1.22772E-04 0.00924 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.49323E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.49186E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.49323E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04478E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05123E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05014E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30015E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30126E-02 0.00101 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54049E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54154E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44000E-03 0.00547  2.68966E-04 0.02031  8.55777E-04 0.01089  6.17857E-04 0.01312  1.36295E-03 0.00905  2.82119E-04 0.01862  5.23309E-05 0.04668 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.46357E-01 0.01583  1.23749E-02 0.00402  3.23276E-02 0.00010  1.06063E-01 0.00057  2.97898E-01 0.00027  1.22964E+00 0.00349  3.80221E+00 0.04526 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25443E-03 0.00806  2.47509E-04 0.03166  8.16657E-04 0.01656  5.91929E-04 0.01972  1.29226E-03 0.01345  2.59983E-04 0.02738  4.60881E-05 0.06333 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.40598E-01 0.02108  1.24749E-02 4.2E-05  3.23322E-02 0.00014  1.06053E-01 0.00076  2.97773E-01 0.00039  1.23702E+00 0.00040  6.38096E+00 0.02827 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.83515E-07 0.00129  8.83158E-07 0.00130  9.92604E-07 0.01803 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.38693E-07 0.00122  8.38354E-07 0.00123  9.42057E-07 0.01798 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22859E-03 0.00940  2.46120E-04 0.03457  8.02468E-04 0.01876  5.81423E-04 0.02067  1.27512E-03 0.01495  2.70105E-04 0.03098  5.33479E-05 0.07250 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53810E-01 0.02557  1.24741E-02 6.2E-05  3.23243E-02 0.00017  1.06149E-01 0.00104  2.97636E-01 0.00046  1.23740E+00 0.00061  6.28695E+00 0.04092 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.78295E-07 0.01540  7.78079E-07 0.01540  8.55340E-07 0.05622 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.38973E-07 0.01538  7.38769E-07 0.01538  8.12167E-07 0.05621 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83454E-03 0.03934  2.78508E-04 0.13091  6.88075E-04 0.07832  5.33383E-04 0.08919  1.05303E-03 0.06075  2.45120E-04 0.11952  3.64289E-05 0.30987 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.10156E-01 0.06307  1.24730E-02 0.00018  3.23274E-02 0.00053  1.06009E-01 0.00309  2.97535E-01 0.00150  1.23712E+00 0.00163  5.72758E+00 0.13307 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81606E-03 0.03888  2.77296E-04 0.12973  6.83576E-04 0.07755  5.19404E-04 0.08872  1.05078E-03 0.06097  2.46359E-04 0.11730  3.86385E-05 0.29138 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.17156E-01 0.06544  1.24730E-02 0.00018  3.23279E-02 0.00053  1.06012E-01 0.00309  2.97538E-01 0.00150  1.23707E+00 0.00162  5.72758E+00 0.13307 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65853E+03 0.03644 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.77234E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.32733E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24472E-03 0.00568 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.69890E+03 0.00565 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92497E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.61086E-05 0.01038  6.61253E-05 0.01037  4.16242E-07 0.50774 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.92745E-05 0.02560  5.93142E-05 0.02558  3.06229E-07 0.69402 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.87089E-04 0.02339  1.87241E-04 0.02333  1.42416E-04 0.50786 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69901E+01 0.01112 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50909E+01 0.00027  4.08134E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 06:08:37 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01096E+00  9.97631E-01  9.99231E-01  1.00279E+00  9.99615E-01  9.95900E-01  9.96520E-01  1.00119E+00  1.00288E+00  1.00951E+00  9.98161E-01  9.91711E-01  1.00397E+00  9.99795E-01  9.95165E-01  1.00034E+00  9.99427E-01  9.95197E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76628E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72337E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39614E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53313E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59110E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53668E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52757E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.28709E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  9.98201E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000845 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00041E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00041E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.95542E+03 ;
RUNNING_TIME              (idx, 1)        =  1.97614E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.89950E-01  9.82833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.95643E+02  3.13764E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.01483E-01  1.80167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.91100E-01  6.15167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.97610E+02  4.44531E+02 ];
CPU_USAGE                 (idx, 1)        = 9.89515 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98318E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90505E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.59165E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66624E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.09744E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95834E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36197E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.60104E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49777E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26091E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.90853E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.63886E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.49493E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.92519E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.05815E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06286E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.29782E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29539E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49056E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.83704E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.77991E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09694E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53710E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.80105E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.80465E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.26516E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43282E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.87372E-03 -1.29690E+27  2.22093E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.76674E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.61541E+18 0.00395  1.72283E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  9.03715E+19 0.00050  9.63839E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  8.15314E+17 0.00560  8.69554E-03 0.00558 ];
PU239_FISS                (idx, [1:   4]) = [  1.02076E+14 0.49851  1.08398E-06 0.49855 ];
TH232_CAPT                (idx, [1:   4]) = [  9.92663E+19 0.00051  7.27402E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31895E+19 0.00140  9.66495E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  2.88912E+17 0.00913  2.11707E-03 0.00911 ];
PU239_CAPT                (idx, [1:   4]) = [  2.52540E+13 1.00000  1.85099E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.63881E+17 0.01267  1.20073E-03 0.01264 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002041 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14659E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002041 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5388164 5.39333E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3702187 3.70560E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911690 9.12530E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002041 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.45058E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36357E+20 3.9E-06  2.36357E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37627E+19 1.2E-07  9.37627E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36514E+20 0.00029  1.21978E+20 0.00020  1.45358E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30276E+20 0.00017  2.15741E+20 0.00011  1.45358E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53031E+20 0.00035  2.53031E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.37978E+22 0.00025  3.20453E+22 0.00022  1.75244E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.30914E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53368E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14685E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50084E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50084E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.03757E-01 0.05026 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21466E-01 0.01370 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.82199E-04 0.02562 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.03603E+03 0.02074 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08750E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.44626E-01 0.04216 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.94957E-01 0.04216 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52080E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.34229E-01 0.00043  3.72414E-02 0.00042  1.22709E-04 0.00942 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.33957E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.34158E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.33957E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02772E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04977E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04872E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30201E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30305E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.57848E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.58195E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.55015E-03 0.00592  2.84629E-04 0.02048  8.74957E-04 0.01142  6.51433E-04 0.01253  1.38004E-03 0.00927  2.97389E-04 0.01830  6.17079E-05 0.04271 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.58184E-01 0.01499  1.23507E-02 0.00450  3.23182E-02 9.9E-05  1.06260E-01 0.00059  2.97975E-01 0.00027  1.23365E+00 0.00286  4.36012E+00 0.03943 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29425E-03 0.00817  2.62883E-04 0.02978  8.18456E-04 0.01619  6.07797E-04 0.01807  1.26950E-03 0.01284  2.76732E-04 0.02965  5.88781E-05 0.05928 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.57679E-01 0.02045  1.24754E-02 4.2E-05  3.23185E-02 0.00014  1.06247E-01 0.00081  2.97962E-01 0.00038  1.23822E+00 0.00049  6.33332E+00 0.02606 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.01332E-07 0.00116  9.01027E-07 0.00116  9.93312E-07 0.01762 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.41991E-07 0.00111  8.41705E-07 0.00112  9.28117E-07 0.01768 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30432E-03 0.00951  2.58170E-04 0.03381  8.34093E-04 0.01917  6.09251E-04 0.02089  1.27419E-03 0.01473  2.66993E-04 0.03247  6.16261E-05 0.06657 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.60455E-01 0.02395  1.24760E-02 4.8E-05  3.23117E-02 0.00018  1.06231E-01 0.00108  2.97913E-01 0.00047  1.23800E+00 0.00070  6.55453E+00 0.03583 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.02370E-07 0.01535  8.01983E-07 0.01536  8.45767E-07 0.05095 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.49668E-07 0.01535  7.49308E-07 0.01536  7.89976E-07 0.05089 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16869E-03 0.04094  2.04306E-04 0.13788  7.76141E-04 0.07935  5.20114E-04 0.09138  1.35154E-03 0.06373  2.71690E-04 0.14061  4.48958E-05 0.31005 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.25953E-01 0.07953  1.24771E-02 0.00012  3.23073E-02 0.00053  1.06324E-01 0.00338  2.98786E-01 0.00175  1.23916E+00 0.00187  6.49112E+00 0.11623 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.21222E-03 0.04040  2.16716E-04 0.13078  7.90543E-04 0.07810  5.26241E-04 0.08920  1.35461E-03 0.06315  2.79419E-04 0.14029  4.46963E-05 0.29552 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.23276E-01 0.07476  1.24771E-02 0.00012  3.23073E-02 0.00053  1.06333E-01 0.00338  2.98782E-01 0.00174  1.23909E+00 0.00187  6.48402E+00 0.11613 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.97189E+03 0.03831 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.96819E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.37763E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38008E-03 0.00638 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77002E+03 0.00645 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.94437E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.55325E-05 0.01137  6.55331E-05 0.01138  7.91543E-07 0.41419 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.62397E-05 0.02985  5.61697E-05 0.02986  1.29454E-06 0.47343 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77293E-04 0.02498  1.77318E-04 0.02492  1.61194E-04 0.40892 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69700E+01 0.01160 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52757E+01 0.00027  4.08694E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 06:40:30 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01051E+00  1.00051E+00  9.95635E-01  1.00669E+00  9.93186E-01  1.00512E+00  1.00467E+00  9.91414E-01  1.00096E+00  9.96590E-01  9.95341E-01  9.92329E-01  1.00010E+00  1.00303E+00  9.91586E-01  1.00790E+00  9.95162E-01  1.00927E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.76866E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72313E-01 9.1E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39107E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52831E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59403E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54179E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53265E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.31396E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.00128E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000460 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00027E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00027E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.27250E+03 ;
RUNNING_TIME              (idx, 1)        =  2.29503E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.91717E-01  1.01767E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.27383E+02  3.17400E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.18717E-01  1.72333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.56067E-01  6.48667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.29499E+02  4.49967E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90184 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98386E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91225E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.67308E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67832E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.01648E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99503E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38209E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.64516E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50729E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.74363E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.20887E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.06772E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.77999E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.20718E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.32769E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32727E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.28968E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.28567E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.47920E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.54738E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27755E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13604E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.63736E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.83013E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.90186E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.29108E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.20437E-02 -2.65921E+27  2.23456E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01457E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.63522E+18 0.00406  1.74139E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  8.82511E+19 0.00055  9.39849E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.29948E+18 0.00327  2.44900E-02 0.00325 ];
PU239_FISS                (idx, [1:   4]) = [  9.00219E+14 0.16806  9.58770E-06 0.16796 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00673E+20 0.00052  7.14053E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28944E+19 0.00139  9.14593E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  8.23927E+17 0.00565  5.84427E-03 0.00565 ];
U238_CAPT                 (idx, [1:   4]) = [  2.62693E+13 1.00000  1.88484E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  4.91186E+14 0.22526  3.48207E-06 0.22525 ];
PU240_CAPT                (idx, [1:   4]) = [  7.66648E+13 0.57620  5.43948E-07 0.57622 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71010E+17 0.01256  1.21286E-03 0.01255 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001336 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15524E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001336 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5454269 5.46012E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3632911 3.63649E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 914156 9.14942E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001336 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.13621E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36260E+20 4.0E-06  2.36260E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37409E+19 1.2E-07  9.37409E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40957E+20 0.00028  1.26135E+20 0.00020  1.48213E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34697E+20 0.00017  2.19876E+20 0.00012  1.48213E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58216E+20 0.00035  2.58216E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45502E+22 0.00025  3.27648E+22 0.00022  1.78537E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36266E+19 0.00150 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58324E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17167E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.48232E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.48232E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.86805E-01 0.05166 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21106E-01 0.01322 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90723E-04 0.02443 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.07200E+03 0.01939 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08509E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.40906E-01 0.04166 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.91458E-01 0.04166 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52036E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99748E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.16464E-01 0.00046  3.65401E-02 0.00045  1.21021E-04 0.00908 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.15650E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.15029E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.15650E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00794E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02204E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02037E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33866E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34054E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.68019E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.68187E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.68703E-03 0.00565  2.82296E-04 0.01874  8.95123E-04 0.01132  6.80507E-04 0.01333  1.44481E-03 0.00894  3.20325E-04 0.01997  6.39648E-05 0.04157 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.69360E-01 0.01460  1.24005E-02 0.00348  3.22928E-02 0.00011  1.06312E-01 0.00056  2.98845E-01 0.00029  1.24308E+00 0.00209  4.58374E+00 0.03788 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.30080E-03 0.00839  2.56336E-04 0.02837  8.03050E-04 0.01681  6.15131E-04 0.01909  1.29178E-03 0.01322  2.77629E-04 0.02837  5.68775E-05 0.06072 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.65510E-01 0.02123  1.24761E-02 3.5E-05  3.22874E-02 0.00014  1.06304E-01 0.00074  2.98950E-01 0.00040  1.24540E+00 0.00083  6.73581E+00 0.02389 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.13240E-07 0.00129  9.12883E-07 0.00129  1.01410E-06 0.01752 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.36870E-07 0.00122  8.36542E-07 0.00122  9.29446E-07 0.01754 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.29438E-03 0.00918  2.51710E-04 0.03366  8.03069E-04 0.01813  6.17447E-04 0.02031  1.29453E-03 0.01453  2.74300E-04 0.03181  5.33262E-05 0.07242 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.59374E-01 0.02546  1.24760E-02 4.8E-05  3.22917E-02 0.00021  1.06298E-01 0.00097  2.98833E-01 0.00052  1.24414E+00 0.00094  7.00543E+00 0.03582 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.03806E-07 0.01532  8.03654E-07 0.01532  7.97014E-07 0.05453 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.36793E-07 0.01531  7.36653E-07 0.01531  7.30803E-07 0.05459 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.02583E-03 0.04213  1.85935E-04 0.14620  7.49462E-04 0.08283  5.43895E-04 0.09374  1.19683E-03 0.06012  3.05525E-04 0.13084  4.41830E-05 0.35077 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.82911E-01 0.10435  1.24759E-02 0.00015  3.23203E-02 0.00068  1.05783E-01 0.00258  2.98688E-01 0.00160  1.24335E+00 0.00247  7.92702E+00 0.10283 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.02720E-03 0.04185  1.94809E-04 0.15118  7.62447E-04 0.08138  5.42679E-04 0.09268  1.17659E-03 0.05848  3.05599E-04 0.12981  4.50782E-05 0.35627 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.82889E-01 0.10392  1.24759E-02 0.00015  3.23212E-02 0.00068  1.05776E-01 0.00258  2.98650E-01 0.00160  1.24336E+00 0.00247  7.91433E+00 0.10282 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79470E+03 0.03955 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.04838E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.29159E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.33872E-03 0.00557 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.69154E+03 0.00568 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95824E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.62980E-05 0.00986  6.63256E-05 0.00998  1.15469E-06 0.34132 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.36507E-05 0.02704  5.37070E-05 0.02706  1.07301E-06 0.43645 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84979E-04 0.02311  1.84764E-04 0.02323  2.70970E-04 0.33342 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68856E+01 0.01155 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53265E+01 0.00026  4.06835E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 07:12:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02735E+00  9.90720E-01  9.91201E-01  9.97513E-01  9.94010E-01  9.96231E-01  9.98828E-01  1.00858E+00  9.98754E-01  1.00757E+00  1.00251E+00  9.91185E-01  9.98346E-01  1.00883E+00  9.90393E-01  1.00196E+00  9.94402E-01  1.00161E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.77657E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72234E-01 9.0E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39007E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52766E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58959E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52335E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51416E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.28239E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.00188E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000534 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00032E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00032E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.59317E+03 ;
RUNNING_TIME              (idx, 1)        =  2.61766E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.00817E-01  1.09100E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.59482E+02  3.20981E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.36917E-01  1.82000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.22350E-01  6.62000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.61762E+02  4.52965E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90645 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98409E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91743E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.86299E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.72771E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.56352E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01669E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39146E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.81240E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55478E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.12015E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.76752E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.04084E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.19583E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04418E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84101E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.37880E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37196E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34273E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56060E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.83261E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30255E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.25451E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.90721E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.42455E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.08841E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.30064E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57630E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.27778E-02 -5.02925E+27  2.25826E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02988E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.59741E+18 0.00402  1.70541E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  8.46194E+19 0.00057  9.03447E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.70560E+18 0.00241  5.02406E-02 0.00236 ];
U238_FISS                 (idx, [1:   4]) = [  2.58440E+13 1.00000  2.76702E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  2.02117E+16 0.03590  2.15586E-04 0.03591 ];
PU240_FISS                (idx, [1:   4]) = [  4.66860E+14 0.23169  4.98347E-06 0.23168 ];
PU241_FISS                (idx, [1:   4]) = [  6.24535E+14 0.20791  6.68115E-06 0.20803 ];
TH232_CAPT                (idx, [1:   4]) = [  9.88970E+19 0.00053  6.92412E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23808E+19 0.00136  8.66864E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.66693E+18 0.00406  1.16707E-02 0.00403 ];
U238_CAPT                 (idx, [1:   4]) = [  1.03160E+14 0.49856  7.23016E-07 0.49857 ];
PU239_CAPT                (idx, [1:   4]) = [  1.06363E+16 0.04768  7.44634E-05 0.04767 ];
PU240_CAPT                (idx, [1:   4]) = [  1.68780E+15 0.12550  1.18298E-05 0.12549 ];
PU241_CAPT                (idx, [1:   4]) = [  1.57707E+14 0.40621  1.10265E-06 0.40621 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78874E+17 0.01225  1.25259E-03 0.01228 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001624 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10567E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001624 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5485541 5.49082E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3597332 3.60067E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 918751 9.19567E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001624 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.12924E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36132E+20 4.1E-06  2.36132E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37025E+19 1.1E-07  9.37025E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42803E+20 0.00029  1.27908E+20 0.00021  1.48953E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36505E+20 0.00018  2.21610E+20 0.00012  1.48953E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60127E+20 0.00033  2.60127E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47082E+22 0.00024  3.29170E+22 0.00021  1.79115E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39214E+19 0.00150 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60427E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17548E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42914E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42914E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.33967E-01 0.05402 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.42260E-01 0.01219 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.83982E-04 0.02435 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.99440E+03 0.01951 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08047E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.11868E-01 0.04371 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.64915E-01 0.04371 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52001E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99829E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07202E-01 0.00046  3.61737E-02 0.00046  1.21848E-04 0.00910 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07746E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.07805E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07746E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99669E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.96352E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.96554E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.41928E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.41607E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.82074E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81714E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.83169E-03 0.00540  2.88625E-04 0.02033  9.06213E-04 0.01094  6.96603E-04 0.01279  1.53086E-03 0.00858  3.36202E-04 0.01782  7.31898E-05 0.03906 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.86909E-01 0.01534  1.24011E-02 0.00348  3.22597E-02 0.00013  1.06580E-01 0.00059  3.00087E-01 0.00031  1.24816E+00 0.00293  4.92022E+00 0.03389 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.41514E-03 0.00809  2.63421E-04 0.02830  7.98384E-04 0.01611  6.23136E-04 0.01849  1.36855E-03 0.01328  2.99187E-04 0.02397  6.24626E-05 0.05766 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.86405E-01 0.02215  1.24756E-02 4.3E-05  3.22639E-02 0.00018  1.06730E-01 0.00089  2.99940E-01 0.00042  1.25392E+00 0.00092  6.74379E+00 0.02158 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.99702E-07 0.00121  8.99322E-07 0.00122  1.01032E-06 0.01766 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16134E-07 0.00114  8.15789E-07 0.00115  9.16534E-07 0.01766 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.36078E-03 0.00916  2.50634E-04 0.03323  7.92273E-04 0.01862  6.06932E-04 0.02213  1.34499E-03 0.01465  3.02772E-04 0.03146  6.31813E-05 0.06911 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.94568E-01 0.02601  1.24762E-02 5.0E-05  3.22606E-02 0.00023  1.06617E-01 0.00117  2.99899E-01 0.00054  1.25280E+00 0.00120  7.12030E+00 0.03104 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.90328E-07 0.01534  7.90314E-07 0.01535  7.11801E-07 0.05631 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.16984E-07 0.01534  7.16971E-07 0.01534  6.45909E-07 0.05635 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.82296E-03 0.04354  1.88478E-04 0.14437  6.71189E-04 0.07981  5.47570E-04 0.09695  1.16660E-03 0.06698  2.04381E-04 0.16507  4.47420E-05 0.26441 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.55778E-01 0.08477  1.24723E-02 0.00021  3.22486E-02 0.00072  1.06520E-01 0.00327  2.99833E-01 0.00183  1.25341E+00 0.00416  7.30151E+00 0.08061 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.84355E-03 0.04295  1.88184E-04 0.13721  6.86339E-04 0.07863  5.45232E-04 0.09567  1.16027E-03 0.06647  2.15369E-04 0.16348  4.81587E-05 0.25462 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.66330E-01 0.08563  1.24724E-02 0.00020  3.22462E-02 0.00071  1.06535E-01 0.00329  2.99898E-01 0.00183  1.25336E+00 0.00416  7.30151E+00 0.08061 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.59088E+03 0.04153 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.90564E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.07843E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32989E-03 0.00638 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.73958E+03 0.00640 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93225E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.65377E-05 0.00961  6.64082E-05 0.00984  6.73287E-07 0.45255 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.68977E-05 0.02895  5.69181E-05 0.02899  3.68876E-07 0.54054 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77767E-04 0.02378  1.77958E-04 0.02382  1.27580E-04 0.44889 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65382E+01 0.01186 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51416E+01 0.00027  4.01589E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 07:45:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01259E+00  1.00318E+00  9.92397E-01  1.00544E+00  9.96677E-01  9.98628E-01  9.97134E-01  1.00249E+00  1.01090E+00  9.94406E-01  1.00000E+00  1.00020E+00  1.00565E+00  9.98669E-01  9.91286E-01  1.00221E+00  9.91760E-01  9.96374E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.78169E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72183E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39045E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52825E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59159E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51441E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50526E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26393E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.00215E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000024 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00023E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00023E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.91774E+03 ;
RUNNING_TIME              (idx, 1)        =  2.94421E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.01008E+00  1.09283E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.91970E+02  3.24886E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.54817E-01  1.79000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.87500E-01  6.50667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.94417E+02  4.55543E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91009 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98417E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92156E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.90524E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73867E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.64868E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01507E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38977E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85603E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56572E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.07030E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98901E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.89587E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.39786E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13440E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04133E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81572E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38574E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35179E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57376E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88562E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73302E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27799E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97483E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.86168E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.12342E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.30070E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21958E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.78803E-02 -6.15588E+27  2.26952E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02736E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.58673E+18 0.00409  1.69296E-02 0.00400 ];
U233_FISS                 (idx, [1:   4]) = [  8.36272E+19 0.00054  8.92358E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.40442E+18 0.00225  5.76685E-02 0.00219 ];
U238_FISS                 (idx, [1:   4]) = [  2.55514E+13 1.00000  2.76243E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  4.99106E+16 0.02262  5.32353E-04 0.02258 ];
PU240_FISS                (idx, [1:   4]) = [  1.81779E+15 0.12152  1.94124E-05 0.12154 ];
PU241_FISS                (idx, [1:   4]) = [  2.08472E+15 0.10566  2.21920E-05 0.10560 ];
TH232_CAPT                (idx, [1:   4]) = [  9.77187E+19 0.00053  6.83824E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22212E+19 0.00147  8.55250E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  1.91763E+18 0.00365  1.34194E-02 0.00363 ];
U238_CAPT                 (idx, [1:   4]) = [  1.54232E+14 0.40622  1.08787E-06 0.40622 ];
PU239_CAPT                (idx, [1:   4]) = [  2.15519E+16 0.03452  1.50900E-04 0.03455 ];
PU240_CAPT                (idx, [1:   4]) = [  4.76542E+15 0.07086  3.33289E-05 0.07090 ];
PU241_CAPT                (idx, [1:   4]) = [  4.43659E+14 0.23865  3.10380E-06 0.23865 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80275E+17 0.01221  1.26154E-03 0.01221 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001157 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10859E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001157 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487598 5.49324E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3599068 3.60253E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 914491 9.15313E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001157 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.34110E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36122E+20 4.2E-06  2.36122E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36882E+19 1.1E-07  9.36882E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42754E+20 0.00027  1.27906E+20 0.00019  1.48481E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36442E+20 0.00016  2.21594E+20 0.00011  1.48481E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60141E+20 0.00035  2.60141E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46737E+22 0.00023  3.28878E+22 0.00021  1.78592E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.38123E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60254E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17323E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.40030E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.40030E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.80024E-01 0.05828 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.44907E-01 0.01203 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.84736E-04 0.02538 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.91439E+03 0.02102 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08472E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.71439E-01 0.04735 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.28275E-01 0.04735 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52029E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99860E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07721E-01 0.00046  3.61931E-02 0.00045  1.24315E-04 0.00898 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08305E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.07725E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08305E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99888E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94054E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94026E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.45231E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.45232E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86403E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86751E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.88385E-03 0.00515  2.89781E-04 0.01968  9.15879E-04 0.01046  6.94904E-04 0.01226  1.56439E-03 0.00814  3.45594E-04 0.01788  7.32980E-05 0.03940 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.90357E-01 0.01446  1.24011E-02 0.00348  3.22569E-02 0.00013  1.06611E-01 0.00054  3.00305E-01 0.00030  1.25328E+00 0.00214  5.08968E+00 0.03330 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47413E-03 0.00784  2.56431E-04 0.02852  8.24076E-04 0.01565  6.00861E-04 0.01861  1.42298E-03 0.01197  3.06684E-04 0.02690  6.30985E-05 0.05737 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.85476E-01 0.02023  1.24759E-02 4.2E-05  3.22529E-02 0.00019  1.06725E-01 0.00080  3.00351E-01 0.00040  1.25508E+00 0.00100  6.91943E+00 0.02086 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.87311E-07 0.00128  8.86885E-07 0.00128  1.00380E-06 0.01880 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.05347E-07 0.00120  8.04960E-07 0.00120  9.11117E-07 0.01877 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42392E-03 0.00897  2.67230E-04 0.03162  8.02590E-04 0.01856  6.09867E-04 0.02239  1.38312E-03 0.01438  3.00138E-04 0.02906  6.09770E-05 0.06976 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.82404E-01 0.02601  1.24760E-02 4.9E-05  3.22474E-02 0.00023  1.06600E-01 0.00106  3.00428E-01 0.00055  1.25726E+00 0.00152  6.84908E+00 0.03363 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.82199E-07 0.01533  7.81878E-07 0.01532  8.48130E-07 0.05469 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.10066E-07 0.01532  7.09774E-07 0.01532  7.70106E-07 0.05467 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08292E-03 0.04335  2.17995E-04 0.15621  7.14969E-04 0.08202  5.80664E-04 0.09189  1.28133E-03 0.06582  2.47806E-04 0.13185  4.01605E-05 0.31917 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.37453E-01 0.09170  1.24792E-02 8.7E-05  3.22889E-02 0.00063  1.05942E-01 0.00241  3.00542E-01 0.00185  1.25741E+00 0.00383  6.65808E+00 0.11868 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08743E-03 0.04276  2.14710E-04 0.15078  7.21111E-04 0.07954  5.79030E-04 0.09049  1.27643E-03 0.06478  2.52724E-04 0.13217  4.34230E-05 0.30888 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.39782E-01 0.09200  1.24792E-02 8.7E-05  3.22886E-02 0.00063  1.05923E-01 0.00239  3.00473E-01 0.00183  1.25738E+00 0.00382  6.65429E+00 0.11874 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96205E+03 0.04098 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.80415E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.99093E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42496E-03 0.00585 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.89018E+03 0.00582 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91331E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.58506E-05 0.01027  6.58324E-05 0.01025  1.49205E-06 0.33361 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.48175E-05 0.02765  5.50725E-05 0.02790  8.08041E-07 0.39970 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79937E-04 0.02434  1.79639E-04 0.02449  2.63176E-04 0.31430 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63151E+01 0.01117 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50526E+01 0.00026  3.99487E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 08:18:16 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01386E+00  1.00106E+00  9.94506E-01  1.00150E+00  9.98581E-01  1.00715E+00  9.87386E-01  1.00282E+00  1.00355E+00  1.00085E+00  1.00062E+00  9.91893E-01  1.00055E+00  1.00626E+00  9.97389E-01  1.00297E+00  9.93869E-01  9.95184E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.78399E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72160E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38998E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52783E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59182E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50918E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50004E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.25586E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.00309E+00 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000094 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00046E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00046E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.24418E+03 ;
RUNNING_TIME              (idx, 1)        =  3.27268E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.11995E+00  1.09867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.24644E+02  3.26734E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.73200E-01  1.83833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.55433E-01  6.78500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.27264E+02  4.57731E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91291 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98452E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92469E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.90545E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73940E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.52293E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00029E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37946E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87099E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56745E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.15333E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17040E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.89876E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.56827E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21144E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20507E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20102E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37759E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34446E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56434E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.91264E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11551E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27052E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99917E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.20564E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10332E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.30015E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.30337E-02 -7.29372E+27  2.28090E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02601E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.56664E+18 0.00421  1.67195E-02 0.00417 ];
U233_FISS                 (idx, [1:   4]) = [  8.27820E+19 0.00055  8.83495E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.90228E+18 0.00215  6.29896E-02 0.00204 ];
U238_FISS                 (idx, [1:   4]) = [  2.52789E+13 1.00000  2.75596E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  8.75877E+16 0.01654  9.34411E-04 0.01647 ];
PU240_FISS                (idx, [1:   4]) = [  3.69619E+15 0.08139  3.94185E-05 0.08137 ];
PU241_FISS                (idx, [1:   4]) = [  4.51505E+15 0.07793  4.81924E-05 0.07790 ];
TH232_CAPT                (idx, [1:   4]) = [  9.67721E+19 0.00053  6.77457E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20611E+19 0.00150  8.44359E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.08794E+18 0.00371  1.46173E-02 0.00370 ];
U238_CAPT                 (idx, [1:   4]) = [  3.62971E+14 0.26379  2.53674E-06 0.26379 ];
PU239_CAPT                (idx, [1:   4]) = [  4.12070E+16 0.02460  2.88421E-04 0.02460 ];
PU240_CAPT                (idx, [1:   4]) = [  1.02853E+16 0.04688  7.20451E-05 0.04690 ];
PU241_CAPT                (idx, [1:   4]) = [  1.01385E+15 0.15394  7.10409E-06 0.15393 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75066E+17 0.01233  1.22559E-03 0.01231 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002285 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11011E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002285 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5488417 5.49349E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3600376 3.60341E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 913492 9.14204E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002285 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.08616E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36132E+20 4.4E-06  2.36132E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36756E+19 1.1E-07  9.36756E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42883E+20 0.00028  1.27996E+20 0.00020  1.48867E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36559E+20 0.00017  2.21672E+20 0.00012  1.48867E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60031E+20 0.00036  2.60031E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46396E+22 0.00025  3.28495E+22 0.00023  1.79012E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37737E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60332E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17138E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.37209E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.37209E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.93195E-01 0.04983 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.00838E-01 0.01421 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.92583E-04 0.02431 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.88198E+03 0.01952 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08582E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.46012E-01 0.04082 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.96068E-01 0.04083 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52074E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99887E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.08281E-01 0.00044  3.62086E-02 0.00043  1.24679E-04 0.00921 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08067E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08150E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08067E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99431E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92442E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92436E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47596E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47565E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.89821E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90306E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.89833E-03 0.00542  2.91125E-04 0.01940  9.13763E-04 0.01097  6.99208E-04 0.01272  1.56921E-03 0.00813  3.54614E-04 0.01760  7.04110E-05 0.03882 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.87171E-01 0.01397  1.24277E-02 0.00284  3.22425E-02 0.00013  1.06581E-01 0.00058  3.00519E-01 0.00031  1.25837E+00 0.00080  4.85476E+00 0.03445 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46528E-03 0.00809  2.62381E-04 0.02973  8.21240E-04 0.01627  6.19049E-04 0.01854  1.39088E-03 0.01211  3.12281E-04 0.02551  5.94426E-05 0.05827 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.77395E-01 0.01951  1.24781E-02 7.1E-05  3.22409E-02 0.00018  1.06583E-01 0.00073  3.00364E-01 0.00041  1.25830E+00 0.00104  6.72109E+00 0.02147 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80184E-07 0.00139  8.79783E-07 0.00139  9.93385E-07 0.02040 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.99346E-07 0.00126  7.98982E-07 0.00126  9.01945E-07 0.02028 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43779E-03 0.00937  2.70241E-04 0.03149  8.06494E-04 0.01848  6.21503E-04 0.02185  1.36911E-03 0.01369  3.05905E-04 0.02959  6.45350E-05 0.06743 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.89393E-01 0.02446  1.24760E-02 5.7E-05  3.22300E-02 0.00025  1.06736E-01 0.00114  3.00592E-01 0.00053  1.25787E+00 0.00144  6.95841E+00 0.03012 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.77282E-07 0.01536  7.76965E-07 0.01537  8.33212E-07 0.05281 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.05994E-07 0.01534  7.05705E-07 0.01535  7.56461E-07 0.05269 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13734E-03 0.04165  3.13292E-04 0.14669  6.51293E-04 0.08014  5.57431E-04 0.09156  1.28776E-03 0.06129  2.77456E-04 0.13004  5.01100E-05 0.25357 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.64799E-01 0.07350  1.24782E-02 0.00012  3.22475E-02 0.00082  1.06402E-01 0.00309  3.01680E-01 0.00192  1.25897E+00 0.00391  7.56425E+00 0.09169 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.12180E-03 0.04119  3.07312E-04 0.14037  6.47230E-04 0.07970  5.57149E-04 0.09051  1.27685E-03 0.06084  2.81810E-04 0.12742  5.14537E-05 0.24512 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.68162E-01 0.07383  1.24782E-02 0.00012  3.22454E-02 0.00081  1.06419E-01 0.00309  3.01664E-01 0.00193  1.25908E+00 0.00390  7.50382E+00 0.09272 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.08306E+03 0.03917 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.72862E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.92725E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43402E-03 0.00616 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.93596E+03 0.00628 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91585E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.65212E-05 0.01004  6.62323E-05 0.01047  1.17732E-06 0.36491 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.61859E-05 0.02857  5.59364E-05 0.02880  8.54685E-07 0.44939 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85479E-04 0.02340  1.85404E-04 0.02350  1.95447E-04 0.35243 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63332E+01 0.01121 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50004E+01 0.00027  3.97823E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 08:51:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01137E+00  1.00023E+00  9.98202E-01  1.00403E+00  9.99460E-01  9.95287E-01  9.90731E-01  1.00355E+00  1.00665E+00  1.00433E+00  9.93932E-01  9.98031E-01  1.00115E+00  1.00391E+00  9.88600E-01  1.00300E+00  9.86796E-01  1.01074E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.79901E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72010E-01 9.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38533E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52390E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59755E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49418E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48503E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24249E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.00878E+00 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1999937 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00037E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00037E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.57650E+03 ;
RUNNING_TIME              (idx, 1)        =  3.60700E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.22680E+00  1.06850E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.57906E+02  3.32627E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.89317E-01  1.61167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.18533E-01  6.30167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.60696E+02  4.59085E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91545 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98481E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92747E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.92969E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74602E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.89409E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96653E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35945E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92904E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57613E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.71891E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75324E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.22777E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03949E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44257E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70417E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34234E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37385E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33873E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55819E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95403E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27067E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26873E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05661E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41864E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.08380E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.30304E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43599E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.34653E-02 -1.18050E+28  2.32602E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02314E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.55594E+18 0.00415  1.66289E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  8.05978E+19 0.00055  8.61407E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.90982E+18 0.00201  7.38487E-02 0.00193 ];
U238_FISS                 (idx, [1:   4]) = [  2.52781E+13 1.00000  2.71407E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  3.05311E+17 0.00914  3.26282E-03 0.00911 ];
PU240_FISS                (idx, [1:   4]) = [  2.11321E+16 0.03459  2.25931E-04 0.03458 ];
PU241_FISS                (idx, [1:   4]) = [  3.47844E+16 0.02684  3.71890E-04 0.02689 ];
TH232_CAPT                (idx, [1:   4]) = [  9.45449E+19 0.00054  6.59114E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.18150E+19 0.00151  8.23671E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.43163E+18 0.00331  1.69536E-02 0.00334 ];
U238_CAPT                 (idx, [1:   4]) = [  1.50960E+15 0.13269  1.05206E-05 0.13270 ];
PU239_CAPT                (idx, [1:   4]) = [  1.45230E+17 0.01291  1.01245E-03 0.01289 ];
PU240_CAPT                (idx, [1:   4]) = [  5.74717E+16 0.02019  4.00799E-04 0.02022 ];
PU241_CAPT                (idx, [1:   4]) = [  7.88471E+15 0.05685  5.49794E-05 0.05680 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77785E+17 0.01213  1.23940E-03 0.01212 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001836 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09829E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001836 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498933 5.50418E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3587205 3.59031E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 915698 9.16492E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001836 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.22935E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36287E+20 4.6E-06  2.36287E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36361E+19 1.0E-07  9.36361E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43433E+20 0.00028  1.28456E+20 0.00019  1.49771E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37069E+20 0.00017  2.22092E+20 0.00011  1.49771E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60609E+20 0.00035  2.60609E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47062E+22 0.00025  3.29076E+22 0.00022  1.79860E+21 0.00193 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.38854E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60955E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17006E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26409E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26409E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.18371E-01 0.05035 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.43780E-01 0.01186 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.16000E-04 0.02371 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.66357E+03 0.02173 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08354E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.44939E-01 0.04083 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.95038E-01 0.04083 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52347E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99971E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05857E-01 0.00044  3.61150E-02 0.00044  1.25574E-04 0.00921 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06486E-01 0.00018 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06728E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06486E-01 0.00018 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97925E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.87818E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88056E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.54575E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.54168E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.00846E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.00222E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93916E-03 0.00589  2.79405E-04 0.01953  9.26679E-04 0.01140  7.24673E-04 0.01220  1.57661E-03 0.00863  3.50643E-04 0.01782  8.11492E-05 0.03693 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.06141E-01 0.01511  1.24772E-02 4.5E-05  3.22163E-02 0.00014  1.06778E-01 0.00054  3.01074E-01 0.00030  1.26039E+00 0.00093  5.35621E+00 0.03071 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.50212E-03 0.00810  2.48823E-04 0.02792  8.41010E-04 0.01660  6.25714E-04 0.01798  1.40201E-03 0.01277  3.09096E-04 0.02578  7.54726E-05 0.05660 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09962E-01 0.02218  1.24770E-02 4.6E-05  3.22128E-02 0.00022  1.06658E-01 0.00069  3.01099E-01 0.00043  1.25890E+00 0.00119  6.99079E+00 0.01963 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.65910E-07 0.00135  8.65489E-07 0.00136  9.83822E-07 0.01640 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.84313E-07 0.00127  7.83932E-07 0.00128  8.91047E-07 0.01639 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45791E-03 0.00916  2.55477E-04 0.03268  8.23887E-04 0.01805  6.35577E-04 0.02123  1.37379E-03 0.01459  2.99021E-04 0.02934  7.01613E-05 0.06141 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06204E-01 0.02530  1.24776E-02 0.00010  3.22211E-02 0.00029  1.06709E-01 0.00103  3.01143E-01 0.00055  1.25866E+00 0.00159  7.25223E+00 0.02711 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.68379E-07 0.01554  7.68180E-07 0.01554  7.87066E-07 0.04578 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.96001E-07 0.01552  6.95822E-07 0.01552  7.13025E-07 0.04584 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13394E-03 0.04216  2.31350E-04 0.14088  7.85103E-04 0.08277  5.56205E-04 0.09094  1.18483E-03 0.06445  2.96225E-04 0.14009  8.02272E-05 0.28900 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.34924E-01 0.10878  1.24760E-02 0.00017  3.21919E-02 0.00076  1.07007E-01 0.00313  3.01114E-01 0.00187  1.26651E+00 0.00533  6.96558E+00 0.08237 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06940E-03 0.04137  2.26269E-04 0.13823  7.48909E-04 0.08180  5.48865E-04 0.09208  1.18283E-03 0.06337  2.84421E-04 0.13803  7.81116E-05 0.27914 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.33144E-01 0.10778  1.24760E-02 0.00017  3.21915E-02 0.00077  1.07021E-01 0.00313  3.01121E-01 0.00187  1.26603E+00 0.00531  6.96558E+00 0.08237 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.13761E+03 0.03997 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.59880E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78856E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46389E-03 0.00576 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.02986E+03 0.00585 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92675E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72877E-05 0.00923  6.72895E-05 0.00923  7.35935E-07 0.47228 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.52210E-05 0.02685  5.51820E-05 0.02688  7.43894E-07 0.47652 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.08324E-04 0.02318  2.08652E-04 0.02321  1.29534E-04 0.44641 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61448E+01 0.01150 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48503E+01 0.00028  3.93496E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 09:25:16 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02201E+00  1.01480E+00  9.93581E-01  1.00638E+00  9.96601E-01  1.00071E+00  9.91940E-01  9.99736E-01  9.99263E-01  9.97287E-01  9.99769E-01  9.94177E-01  1.00038E+00  1.00068E+00  9.90723E-01  1.00196E+00  9.92242E-01  9.97761E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.83922E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71608E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37323E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51362E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60683E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49486E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48568E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.28096E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.03399E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000585 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00035E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00035E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.91009E+03 ;
RUNNING_TIME              (idx, 1)        =  3.94265E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.33337E+00  1.06567E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.91297E+02  3.33903E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.07050E-01  1.77333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.79283E-01  6.06833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.94261E+02  4.61035E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91743 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98468E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92977E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.91592E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74053E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.57255E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92388E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33783E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95786E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57276E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.54041E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39320E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36562E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14317E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69285E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23919E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58147E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34687E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31706E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52918E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99017E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59858E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23576E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03573E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.16342E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01959E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.30785E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06796E-01 -2.35801E+28  2.44377E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02047E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.52192E+18 0.00422  1.62571E-02 0.00417 ];
U233_FISS                 (idx, [1:   4]) = [  7.93672E+19 0.00056  8.47833E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.13124E+18 0.00192  7.61772E-02 0.00182 ];
U238_FISS                 (idx, [1:   4]) = [  1.31821E+14 0.44544  1.39880E-06 0.44547 ];
PU239_FISS                (idx, [1:   4]) = [  6.10259E+17 0.00660  6.51970E-03 0.00661 ];
PU240_FISS                (idx, [1:   4]) = [  7.28589E+16 0.01892  7.78123E-04 0.01890 ];
PU241_FISS                (idx, [1:   4]) = [  1.28493E+17 0.01442  1.37245E-03 0.01439 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28626E+19 0.00053  6.43931E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15230E+19 0.00150  7.99028E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51563E+18 0.00301  1.74447E-02 0.00301 ];
U238_CAPT                 (idx, [1:   4]) = [  2.34793E+15 0.10298  1.62843E-05 0.10298 ];
PU239_CAPT                (idx, [1:   4]) = [  2.92257E+17 0.00949  2.02660E-03 0.00948 ];
PU240_CAPT                (idx, [1:   4]) = [  1.93324E+17 0.01193  1.34047E-03 0.01190 ];
PU241_CAPT                (idx, [1:   4]) = [  2.62253E+16 0.03112  1.81789E-04 0.03105 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84346E+17 0.01233  1.27845E-03 0.01234 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001738 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07604E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001738 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5508123 5.51338E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3575832 3.57888E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 917783 9.18498E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001738 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.42727E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36592E+20 4.9E-06  2.36592E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35954E+19 1.0E-07  9.35954E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44273E+20 0.00027  1.28776E+20 0.00019  1.54965E+19 0.00169 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37868E+20 0.00016  2.22372E+20 0.00011  1.54965E+19 0.00169 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61571E+20 0.00036  2.61571E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50478E+22 0.00025  3.32058E+22 0.00022  1.84200E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40269E+19 0.00161 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61895E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17450E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  8.04932E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.04932E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.23058E-01 0.04288 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31745E-01 0.01003 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.71247E-04 0.01913 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.66395E+03 0.01970 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08154E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.53511E-01 0.03242 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.93572E-01 0.03242 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52782E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00058E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04664E-01 0.00044  3.60611E-02 0.00043  1.27237E-04 0.00881 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.04379E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.04565E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.04379E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95832E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85221E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85013E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58642E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58937E-02 0.00100 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07409E-01 0.00139 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07172E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93787E-03 0.00534  3.02440E-04 0.01961  9.02121E-04 0.01119  7.07664E-04 0.01329  1.58494E-03 0.00786  3.61332E-04 0.01769  7.93713E-05 0.03757 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05194E-01 0.01468  1.23046E-02 0.00533  3.21945E-02 0.00016  1.06762E-01 0.00058  3.01324E-01 0.00031  1.25620E+00 0.00226  5.29229E+00 0.03175 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48012E-03 0.00786  2.75712E-04 0.02839  7.85753E-04 0.01589  6.36102E-04 0.01918  1.39396E-03 0.01151  3.18100E-04 0.02594  7.04959E-05 0.05865 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.10283E-01 0.02302  1.24797E-02 0.00013  3.21971E-02 0.00021  1.06784E-01 0.00085  3.01461E-01 0.00043  1.26000E+00 0.00128  7.02934E+00 0.01990 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.62220E-07 0.00131  8.61766E-07 0.00131  9.87852E-07 0.01721 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.79948E-07 0.00124  7.79538E-07 0.00124  8.93554E-07 0.01720 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.50844E-03 0.00887  2.70573E-04 0.03240  8.15736E-04 0.01875  6.17220E-04 0.02123  1.41855E-03 0.01296  3.08459E-04 0.02938  7.78979E-05 0.05854 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.16560E-01 0.02452  1.24778E-02 0.00010  3.21942E-02 0.00027  1.06869E-01 0.00109  3.01373E-01 0.00053  1.25929E+00 0.00177  7.07203E+00 0.02777 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63694E-07 0.01552  7.63476E-07 0.01552  7.84416E-07 0.05005 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.91081E-07 0.01552  6.90883E-07 0.01552  7.09597E-07 0.04999 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.29457E-03 0.04078  2.65021E-04 0.13324  7.69848E-04 0.07541  5.91224E-04 0.09117  1.28497E-03 0.05956  3.25461E-04 0.12454  5.80439E-05 0.29114 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.78736E-01 0.08694  1.24787E-02 9.2E-05  3.21568E-02 0.00084  1.06927E-01 0.00311  3.01861E-01 0.00195  1.25595E+00 0.00472  7.20130E+00 0.08500 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.27782E-03 0.04011  2.68002E-04 0.13061  7.54066E-04 0.07431  5.72504E-04 0.08970  1.29772E-03 0.05920  3.26871E-04 0.12020  5.86616E-05 0.27831 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.88664E-01 0.08808  1.24787E-02 9.2E-05  3.21573E-02 0.00084  1.06968E-01 0.00314  3.01845E-01 0.00194  1.25599E+00 0.00472  7.20248E+00 0.08501 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.36312E+03 0.03807 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.54909E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.73330E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53110E-03 0.00544 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.13183E+03 0.00551 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.05155E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.82904E-05 0.00628  6.80959E-05 0.00661  1.80992E-06 0.28334 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.66234E-05 0.02038  5.65091E-05 0.02044  2.01039E-06 0.36836 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.62106E-04 0.01837  2.61839E-04 0.01852  3.24382E-04 0.27495 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65404E+01 0.01152 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48568E+01 0.00028  3.91049E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 09:58:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00901E+00  1.00776E+00  9.92216E-01  1.00138E+00  9.99917E-01  1.00612E+00  9.96405E-01  9.92640E-01  9.97165E-01  1.00288E+00  9.96781E-01  9.89186E-01  1.00492E+00  1.00055E+00  1.00006E+00  1.00588E+00  9.89856E-01  1.00727E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.87932E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71207E-01 9.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35695E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49928E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62539E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49580E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48649E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33476E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.06065E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000229 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00047E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00047E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.24387E+03 ;
RUNNING_TIME              (idx, 1)        =  4.27854E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.44220E+00  1.08833E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.24702E+02  3.34053E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.24917E-01  1.78667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.41083E-01  6.17000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.27851E+02  4.61368E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91895 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98463E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93151E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.91361E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73875E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.41169E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91347E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33275E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96588E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57141E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64118E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56789E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45888E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16093E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76499E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39554E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93216E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33940E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31156E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52148E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00573E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00907E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22620E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02467E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.35372E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00309E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.31007E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93016E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61358E-01 -3.56273E+28  2.56424E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01632E+00 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.51286E+18 0.00410  1.61752E-02 0.00405 ];
U233_FISS                 (idx, [1:   4]) = [  7.91704E+19 0.00058  8.46506E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.11282E+18 0.00187  7.60537E-02 0.00182 ];
U238_FISS                 (idx, [1:   4]) = [  1.31491E+14 0.44548  1.41431E-06 0.44544 ];
PU239_FISS                (idx, [1:   4]) = [  6.53356E+17 0.00640  6.98586E-03 0.00639 ];
PU240_FISS                (idx, [1:   4]) = [  8.08876E+16 0.01804  8.64733E-04 0.01802 ];
PU241_FISS                (idx, [1:   4]) = [  1.54654E+17 0.01313  1.65352E-03 0.01311 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22694E+19 0.00056  6.39212E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15693E+19 0.00154  8.01505E-02 0.00152 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49641E+18 0.00308  1.72941E-02 0.00303 ];
U238_CAPT                 (idx, [1:   4]) = [  3.00850E+15 0.09558  2.08270E-05 0.09557 ];
PU239_CAPT                (idx, [1:   4]) = [  3.10473E+17 0.00919  2.15098E-03 0.00920 ];
PU240_CAPT                (idx, [1:   4]) = [  2.25701E+17 0.01063  1.56370E-03 0.01064 ];
PU241_CAPT                (idx, [1:   4]) = [  3.27814E+16 0.02915  2.27100E-04 0.02913 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83796E+17 0.01203  1.27343E-03 0.01206 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002329 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07830E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002329 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5504389 5.50925E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3566602 3.56954E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931338 9.31996E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002329 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.04308E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36657E+20 4.8E-06  2.36657E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35891E+19 1.1E-07  9.35891E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44258E+20 0.00028  1.28398E+20 0.00019  1.58607E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37847E+20 0.00017  2.21987E+20 0.00011  1.58607E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62014E+20 0.00034  2.62014E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53508E+22 0.00024  3.34884E+22 0.00022  1.86247E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.44211E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62269E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17673E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  7.86236E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.86236E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.54717E-01 0.03859 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24418E-01 0.00911 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.23412E-04 0.01892 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.22913E+03 0.02123 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06805E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.20188E-01 0.02764 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.53045E-01 0.02764 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52868E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.02594E-01 0.00045  3.59818E-02 0.00045  1.22296E-04 0.00887 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.03338E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03276E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.03338E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.96195E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84843E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84861E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59249E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59177E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.08196E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.08086E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.92837E-03 0.00569  2.87290E-04 0.01949  9.26684E-04 0.01104  7.10367E-04 0.01143  1.57495E-03 0.00832  3.47354E-04 0.01810  8.17232E-05 0.03850 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.02907E-01 0.01497  1.23534E-02 0.00450  3.21975E-02 0.00017  1.06761E-01 0.00059  3.01408E-01 0.00031  1.25747E+00 0.00119  5.32865E+00 0.03109 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47979E-03 0.00779  2.55137E-04 0.02956  8.32530E-04 0.01598  6.15178E-04 0.01635  1.39860E-03 0.01211  3.05929E-04 0.02606  7.24217E-05 0.05064 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.01209E-01 0.01984  1.24782E-02 1.0E-04  3.21889E-02 0.00022  1.06686E-01 0.00073  3.01434E-01 0.00043  1.25599E+00 0.00159  6.90065E+00 0.02030 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.69906E-07 0.00143  8.69401E-07 0.00143  1.01964E-06 0.01740 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.85098E-07 0.00136  7.84641E-07 0.00137  9.20380E-07 0.01741 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.38220E-03 0.00904  2.49784E-04 0.03255  7.91236E-04 0.01849  5.93582E-04 0.02022  1.36892E-03 0.01416  3.06367E-04 0.02908  7.23188E-05 0.05896 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04709E-01 0.02375  1.24783E-02 8.7E-05  3.22106E-02 0.00033  1.06906E-01 0.00110  3.01345E-01 0.00054  1.25771E+00 0.00179  6.75017E+00 0.02987 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.66517E-07 0.01551  7.65968E-07 0.01551  8.37366E-07 0.05755 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.92064E-07 0.01551  6.91570E-07 0.01551  7.55565E-07 0.05749 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14927E-03 0.04192  2.48037E-04 0.13426  7.78066E-04 0.08007  5.28568E-04 0.09133  1.29216E-03 0.06430  2.44928E-04 0.15115  5.75137E-05 0.25951 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.61383E-01 0.09127  1.24793E-02 0.00017  3.21552E-02 0.00081  1.06567E-01 0.00294  3.01281E-01 0.00185  1.25992E+00 0.00564  6.85022E+00 0.09695 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.14798E-03 0.04157  2.48111E-04 0.13064  7.69246E-04 0.07853  5.28984E-04 0.08946  1.29510E-03 0.06361  2.47585E-04 0.14907  5.89531E-05 0.25650 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.65226E-01 0.09400  1.24793E-02 0.00017  3.21550E-02 0.00080  1.06568E-01 0.00294  3.01235E-01 0.00184  1.25995E+00 0.00563  6.82542E+00 0.09714 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.13727E+03 0.03942 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.62625E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78528E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46642E-03 0.00575 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.02028E+03 0.00586 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.19604E-09 0.00097 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.95489E-05 0.00597  6.95773E-05 0.00597  1.20372E-06 0.32039 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.12610E-05 0.01911  6.12203E-05 0.01912  1.45834E-06 0.45859 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.12054E-04 0.01830  3.12278E-04 0.01831  2.56142E-04 0.31501 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60575E+01 0.01116 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48649E+01 0.00028  3.90630E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 36])  = '/gv/data/cfx/tiago/doutorado/malha5+' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Fri May  6 02:51:00 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri May  6 10:16:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651816260693 ;
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

MPI_TASKS                 (idx, 1)        = 5 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03783E+00  9.99461E-01  9.89410E-01  9.99991E-01  9.88479E-01  9.97713E-01  9.90667E-01  9.96636E-01  1.00047E+00  1.00365E+00  9.97232E-01  9.98587E-01  1.00612E+00  1.00621E+00  9.85205E-01  1.00674E+00  9.96464E-01  9.99134E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.92329E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70767E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34606E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49043E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63339E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50208E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49266E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.37878E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.08842E+00 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2000543 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.00041E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.00041E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.42151E+03 ;
RUNNING_TIME              (idx, 1)        =  4.45821E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07880E+00  1.07880E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.55033E+00  1.08133E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.42483E+02  1.77815E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.43117E-01  1.82000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.07500E-02  1.07500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.00377E+00  6.25833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.45817E+02  4.45817E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91770 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97537E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93066E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22543.70;
MEMSIZE                   (idx, 1)        = 22327.81;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 176.10;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 215.89;

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

TOT_ACTIVITY              (idx, 1)        =  6.90944E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73809E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.63563E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90629E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32845E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96885E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57112E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65500E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61506E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47046E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16155E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78605E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44184E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03236E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33764E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31031E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51969E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00892E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13553E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22037E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02140E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.38939E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98943E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.31310E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16444E-01 -4.77901E+28  2.68587E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01104E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.49917E+18 0.00425  1.60055E-02 0.00419 ];
U233_FISS                 (idx, [1:   4]) = [  7.92992E+19 0.00053  8.46675E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.09213E+18 0.00197  7.57204E-02 0.00188 ];
U238_FISS                 (idx, [1:   4]) = [  2.36452E+14 0.33068  2.53525E-06 0.33067 ];
PU239_FISS                (idx, [1:   4]) = [  6.59417E+17 0.00631  7.03952E-03 0.00624 ];
PU240_FISS                (idx, [1:   4]) = [  8.78005E+16 0.01694  9.37282E-04 0.01692 ];
PU241_FISS                (idx, [1:   4]) = [  1.55773E+17 0.01313  1.66349E-03 0.01313 ];
TH232_CAPT                (idx, [1:   4]) = [  9.18305E+19 0.00053  6.35685E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15460E+19 0.00141  7.99264E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48566E+18 0.00345  1.72069E-02 0.00343 ];
U238_CAPT                 (idx, [1:   4]) = [  2.91419E+15 0.09121  2.01516E-05 0.09113 ];
PU239_CAPT                (idx, [1:   4]) = [  3.15812E+17 0.00908  2.18635E-03 0.00909 ];
PU240_CAPT                (idx, [1:   4]) = [  2.31978E+17 0.01100  1.60577E-03 0.01099 ];
PU241_CAPT                (idx, [1:   4]) = [  3.45964E+16 0.02909  2.39461E-04 0.02909 ];
SM149_CAPT                (idx, [1:   4]) = [  1.91798E+17 0.01210  1.32778E-03 0.01211 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002037 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08926E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002037 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5495691 5.50075E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3563370 3.56640E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 942976 9.43738E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002037 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.63913E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 2.0E-09  3.52483E-02 2.0E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36668E+20 5.1E-06  2.36668E+20 5.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35884E+19 1.1E-07  9.35884E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44457E+20 0.00029  1.28056E+20 0.00018  1.64006E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38045E+20 0.00018  2.21644E+20 0.00010  1.64006E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62620E+20 0.00035  2.62620E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.56742E+22 0.00027  3.37743E+22 0.00023  1.89983E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.47857E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62831E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.18109E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51104E+04 ;
TOT_FMASS                 (idx, 1)        =  7.70068E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51104E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.70068E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.80971E-01 0.03892 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.99834E-01 0.00877 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.93297E-04 0.01685 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.96631E+03 0.02314 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.05634E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.17041E-01 0.02792 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.49283E-01 0.02792 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52882E+00 5.1E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00073E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.01692E-01 0.00043  3.59515E-02 0.00042  1.24003E-04 0.00895 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.01446E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.01234E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.01446E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95413E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84957E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84936E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59071E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59059E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07839E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.08340E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94866E-03 0.00547  2.83925E-04 0.02065  9.14754E-04 0.01072  6.99717E-04 0.01273  1.59788E-03 0.00905  3.70049E-04 0.01639  8.23306E-05 0.03718 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13486E-01 0.01409  1.24048E-02 0.00348  3.21988E-02 0.00017  1.06901E-01 0.00060  3.01258E-01 0.00031  1.25795E+00 0.00103  5.42298E+00 0.03041 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48876E-03 0.00780  2.46892E-04 0.02837  7.95220E-04 0.01571  6.23350E-04 0.01817  1.42811E-03 0.01270  3.20585E-04 0.02492  7.46017E-05 0.05238 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.16331E-01 0.02047  1.24805E-02 0.00015  3.21958E-02 0.00024  1.06908E-01 0.00084  3.01243E-01 0.00043  1.25808E+00 0.00145  7.02430E+00 0.01939 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.78578E-07 0.00139  8.78148E-07 0.00139  9.99574E-07 0.01599 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.92139E-07 0.00133  7.91751E-07 0.00133  9.01310E-07 0.01601 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44058E-03 0.00899  2.50653E-04 0.03372  7.94675E-04 0.01910  6.08236E-04 0.02216  1.39087E-03 0.01440  3.23245E-04 0.02864  7.29013E-05 0.06319 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14026E-01 0.02592  1.24767E-02 5.6E-05  3.22030E-02 0.00030  1.06968E-01 0.00116  3.01426E-01 0.00053  1.25834E+00 0.00190  6.77334E+00 0.02946 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.68405E-07 0.01556  7.67840E-07 0.01557  8.08016E-07 0.05385 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.93060E-07 0.01555  6.92549E-07 0.01556  7.29340E-07 0.05397 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09730E-03 0.04155  2.43828E-04 0.13461  6.81068E-04 0.08386  5.38980E-04 0.08898  1.21320E-03 0.06338  3.83568E-04 0.11773  3.66558E-05 0.32799 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.65309E-01 0.08402  1.24778E-02 0.00012  3.21565E-02 0.00100  1.06959E-01 0.00326  3.01544E-01 0.00190  1.25469E+00 0.00629  7.26367E+00 0.10720 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09210E-03 0.04157  2.40346E-04 0.13007  6.76709E-04 0.08453  5.29818E-04 0.08816  1.22719E-03 0.06214  3.81628E-04 0.11652  3.64168E-05 0.30872 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.67952E-01 0.08218  1.24778E-02 0.00012  3.21562E-02 0.00099  1.06971E-01 0.00326  3.01528E-01 0.00189  1.25427E+00 0.00626  7.19181E+00 0.10738 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.06901E+03 0.03921 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.68785E-07 0.00081 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.83302E-07 0.00068 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44288E-03 0.00648 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.96408E+03 0.00651 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.34800E-09 0.00102 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.91572E-05 0.00475  6.91786E-05 0.00480  2.48849E-06 0.24107 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.12601E-05 0.01731  6.12499E-05 0.01741  2.16176E-06 0.29612 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.77694E-04 0.01622  3.77378E-04 0.01623  4.53620E-04 0.23313 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60022E+01 0.01130 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49266E+01 0.00027  3.90589E+01 0.00037 ];

