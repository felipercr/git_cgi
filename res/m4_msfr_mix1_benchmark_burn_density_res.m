
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 19 19:45:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03517E+00  1.01250E+00  9.97387E-01  9.91077E-01  1.00648E+00  9.98562E-01  9.87526E-01  1.00371E+00  9.76490E-01  1.01027E+00  9.96407E-01  1.00316E+00  9.87482E-01  9.90098E-01  9.92038E-01  1.00217E+00  9.96040E-01  1.01343E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.12542E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68746E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31414E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45850E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72186E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48640E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47662E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.46725E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38388E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333914 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33395E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33395E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.57688E+02 ;
RUNNING_TIME              (idx, 1)        =  2.85807E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.76100E-01  1.76100E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.74048E+01  2.74048E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.07042E+00  2.02712E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.65940E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.01615 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98519E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.94166E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  2.69001E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.11547E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.05908E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.69001E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.11547E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.58548E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.37245E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.58548E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.37245E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.24190E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.05813E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09307E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.22349E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15175E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.76739E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.56192E+18 0.00422  1.66591E-02 0.00414 ];
U233_FISS                 (idx, [1:   4]) = [  9.21899E+19 0.00047  9.83341E-01 7.0E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.25546E+19 0.00053  7.48091E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33747E+19 0.00134  1.08103E-01 0.00126 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001860 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15103E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001860 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5133129 5.13830E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3890038 3.89366E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 978693 9.79544E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001860 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.28408E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36457E+20 3.8E-06  2.36457E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37769E+19 1.3E-07  9.37769E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23686E+20 0.00032  1.08998E+20 0.00024  1.46880E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17463E+20 0.00018  2.02775E+20 0.00013  1.46880E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40783E+20 0.00037  2.40783E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.31905E+22 0.00026  3.13957E+22 0.00023  1.79478E+21 0.00171 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35867E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41050E+20 0.00023 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07906E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29424E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.22051E-01 0.06660 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.81296E-01 0.00817 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.34420E-04 0.01430 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.17368E+04 0.01925 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02054E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 9.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.44158E-01 0.05783 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.90833E-01 0.05783 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52149E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.81863E-01 0.00042  1.08728E-01 0.00040  3.58961E-04 0.00913 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.82090E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.82075E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.82090E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08874E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02743E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02827E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33115E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32988E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50028E-01 0.00162 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49583E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.31840E-03 0.00552  2.63018E-04 0.02182  8.01580E-04 0.01166  6.08388E-04 0.01229  1.30833E-03 0.00879  2.84355E-04 0.01900  5.27308E-05 0.04699 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50407E-01 0.01535  1.24755E-02 3.2E-05  3.23348E-02 8.8E-05  1.06076E-01 0.00055  2.97409E-01 0.00027  1.23542E+00 0.00023  5.13621E+00 0.03969 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29877E-03 0.00834  2.53048E-04 0.03015  7.85707E-04 0.01643  6.05899E-04 0.01835  1.30643E-03 0.01354  2.95311E-04 0.02890  5.23798E-05 0.06520 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.54400E-01 0.02030  1.24761E-02 3.3E-05  3.23373E-02 0.00013  1.05884E-01 0.00072  2.97405E-01 0.00036  1.23534E+00 0.00031  6.44483E+00 0.02949 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.83885E-07 0.00125  8.83525E-07 0.00125  9.85764E-07 0.01966 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.67807E-07 0.00117  8.67455E-07 0.00117  9.67732E-07 0.01956 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30791E-03 0.00941  2.62150E-04 0.03156  7.87514E-04 0.01800  6.07822E-04 0.02029  1.31549E-03 0.01530  2.84854E-04 0.02926  5.00893E-05 0.07534 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47317E-01 0.02416  1.24756E-02 5.0E-05  3.23370E-02 0.00016  1.06036E-01 0.00098  2.97465E-01 0.00045  1.23530E+00 0.00038  6.05515E+00 0.04382 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.86233E-07 0.01983  7.86009E-07 0.01984  8.60797E-07 0.04859 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.72047E-07 0.01982  7.71826E-07 0.01983  8.45322E-07 0.04855 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98808E-03 0.04250  2.73570E-04 0.12343  7.87216E-04 0.07804  4.76927E-04 0.08507  1.15547E-03 0.06241  2.57145E-04 0.12818  3.77505E-05 0.30764 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.22545E-01 0.08506  1.24765E-02 0.00011  3.23553E-02 0.00060  1.05728E-01 0.00274  2.98279E-01 0.00182  1.23697E+00 0.00100  5.21590E+00 0.14440 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98875E-03 0.04127  2.74369E-04 0.12281  7.90920E-04 0.07588  4.82060E-04 0.08346  1.14897E-03 0.06052  2.50785E-04 0.12741  4.16470E-05 0.31714 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.21237E-01 0.08587  1.24764E-02 0.00012  3.23541E-02 0.00059  1.05683E-01 0.00266  2.98225E-01 0.00180  1.23704E+00 0.00100  5.21590E+00 0.14440 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81949E+03 0.03807 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.78127E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.62154E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34999E-03 0.00552 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81503E+03 0.00550 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.24182E-09 0.00131 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.20742E-05 0.00434  7.20969E-05 0.00435  4.28886E-06 0.23235 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  8.16673E-05 0.01568  8.18129E-05 0.01576  3.12378E-06 0.28679 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.10075E-04 0.01400  5.09868E-04 0.01396  5.80996E-04 0.22371 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70210E+01 0.01243 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47662E+01 0.00027  4.03534E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 19 20:29:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01802E+00  1.00333E+00  1.00292E+00  1.00165E+00  9.97709E-01  9.99545E-01  9.95485E-01  1.00439E+00  9.38179E-01  1.00506E+00  1.00226E+00  1.00916E+00  1.01018E+00  1.00296E+00  9.91713E-01  1.00448E+00  1.00510E+00  1.00786E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.12133E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68787E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31112E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45537E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72314E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48917E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47933E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.48410E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38405E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333893 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33389E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33389E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.68606E+02 ;
RUNNING_TIME              (idx, 1)        =  7.24719E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.95417E-01  1.19317E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.11524E+01  4.37477E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.78667E-02  1.78667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.06192E+00  2.99143E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.95220E+01  4.27568E+02 ];
CPU_USAGE                 (idx, 1)        = 9.22572 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98916E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.16277E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  5.77438E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60381E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.66083E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.48057E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.04349E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26283E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46838E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.45292E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65570E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.39072E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.78669E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06219E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17703E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44218E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35384E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17384E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55156E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60247E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38115E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39507E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51598E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.52703E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.30254E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.29544E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26594E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.11344E-04 -2.39584E+25  2.15198E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.88450E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.56703E+18 0.00393  1.67248E-02 0.00384 ];
U233_FISS                 (idx, [1:   4]) = [  9.20940E+19 0.00049  9.82974E-01 6.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.69565E+14 0.37417  1.80507E-06 0.37418 ];
TH232_CAPT                (idx, [1:   4]) = [  9.36390E+19 0.00055  7.44188E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33690E+19 0.00148  1.06249E-01 0.00141 ];
U235_CAPT                 (idx, [1:   4]) = [  1.95264E+14 0.34941  1.55183E-06 0.34941 ];
SM149_CAPT                (idx, [1:   4]) = [  1.31785E+16 0.04643  1.04771E-04 0.04646 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001673 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14813E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001673 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5168879 5.17422E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3849071 3.85268E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983723 9.84582E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001673 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.96046E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36448E+20 3.6E-06  2.36448E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37765E+19 1.2E-07  9.37765E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25839E+20 0.00032  1.11092E+20 0.00023  1.47468E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19616E+20 0.00019  2.04869E+20 0.00013  1.47468E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43181E+20 0.00036  2.43181E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.35514E+22 0.00025  3.17526E+22 0.00023  1.79872E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39441E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43560E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09047E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29514E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29514E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.62271E-01 0.07213 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.97569E-01 0.00789 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.72175E-04 0.01599 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.06050E+04 0.02665 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01550E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.49461E-01 0.05669 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.95252E-01 0.05669 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52140E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.71435E-01 0.00041  1.07585E-01 0.00041  3.51734E-04 0.00926 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.71921E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.72349E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.71921E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07802E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03028E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03116E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32738E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32605E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50278E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49924E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33711E-03 0.00580  2.74743E-04 0.02068  8.23407E-04 0.01074  6.16830E-04 0.01264  1.29127E-03 0.00905  2.77686E-04 0.02027  5.31664E-05 0.04451 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.46379E-01 0.01523  1.24756E-02 2.8E-05  3.23359E-02 8.7E-05  1.06062E-01 0.00055  2.97210E-01 0.00027  1.23561E+00 0.00021  5.00681E+00 0.03948 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23256E-03 0.00856  2.59072E-04 0.02930  8.06748E-04 0.01661  6.10640E-04 0.01986  1.23407E-03 0.01294  2.71792E-04 0.02946  5.02316E-05 0.06565 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45956E-01 0.02127  1.24755E-02 4.0E-05  3.23388E-02 0.00014  1.06022E-01 0.00082  2.97232E-01 0.00037  1.23554E+00 0.00031  6.29812E+00 0.02927 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.93829E-07 0.00120  8.93526E-07 0.00121  9.89745E-07 0.01630 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.68252E-07 0.00112  8.67957E-07 0.00113  9.61735E-07 0.01636 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.25109E-03 0.00939  2.58916E-04 0.03138  8.07904E-04 0.01809  6.04616E-04 0.02065  1.25445E-03 0.01434  2.73692E-04 0.03175  5.15140E-05 0.07165 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.50977E-01 0.02508  1.24761E-02 4.3E-05  3.23419E-02 0.00016  1.06144E-01 0.00100  2.97255E-01 0.00045  1.23542E+00 0.00042  6.71150E+00 0.03912 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.91767E-07 0.01960  7.91345E-07 0.01960  8.91196E-07 0.05297 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.69173E-07 0.01960  7.68761E-07 0.01960  8.66461E-07 0.05309 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00135E-03 0.04165  2.33969E-04 0.14745  7.72960E-04 0.07083  5.80650E-04 0.10101  1.11805E-03 0.06169  2.23058E-04 0.12516  7.26688E-05 0.27869 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.25092E-01 0.08396  1.24747E-02 0.00016  3.23242E-02 0.00046  1.05981E-01 0.00304  2.97825E-01 0.00164  1.23772E+00 0.00097  5.60108E+00 0.13048 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.02707E-03 0.04074  2.29036E-04 0.14745  7.71217E-04 0.06964  5.78650E-04 0.10043  1.14621E-03 0.06008  2.31481E-04 0.12280  7.04776E-05 0.27952 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.21025E-01 0.07869  1.24746E-02 0.00016  3.23250E-02 0.00046  1.06034E-01 0.00312  2.97738E-01 0.00159  1.23765E+00 0.00096  5.60108E+00 0.13048 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.80254E+03 0.03689 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.87797E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.62387E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32777E-03 0.00564 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.74804E+03 0.00557 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17732E-09 0.00129 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.21846E-05 0.00423  7.21520E-05 0.00422  3.89106E-06 0.26337 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.57750E-05 0.01603  7.57759E-05 0.01608  3.75644E-06 0.31584 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.51157E-04 0.01575  4.51166E-04 0.01575  4.56088E-04 0.25313 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72685E+01 0.01067 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47933E+01 0.00030  4.03916E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 19 21:15:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02240E+00  9.97898E-01  9.97545E-01  1.00363E+00  9.97290E-01  1.00337E+00  9.96918E-01  1.00835E+00  9.36755E-01  1.00392E+00  9.99760E-01  1.00758E+00  1.00288E+00  1.00138E+00  1.00072E+00  1.00761E+00  9.99594E-01  1.01244E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.10280E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68972E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31250E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45597E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72117E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49994E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49009E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.50225E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.37722E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333387 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33342E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33342E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.10478E+03 ;
RUNNING_TIME              (idx, 1)        =  1.19138E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.23967E-01  1.28550E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.17606E+02  4.64536E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.34667E-02  7.56000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.26033E+00  3.19837E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.15982E+02  6.43404E+02 ];
CPU_USAGE                 (idx, 1)        = 9.27313 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98944E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.21785E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.39446E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65454E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.25619E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.86784E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.30647E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49534E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49252E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.76250E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.40155E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.86691E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.01802E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89537E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.59975E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49236E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34327E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33164E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54033E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.70109E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17615E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.01367E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51467E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.35399E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.52669E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.39729E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61905E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.92916E-04 -1.27580E+26  2.15302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.04233E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.59056E+18 0.00383  1.69566E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  9.20728E+19 0.00053  9.81609E-01 6.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.19800E+16 0.04346  1.27809E-04 0.04354 ];
TH232_CAPT                (idx, [1:   4]) = [  9.50516E+19 0.00051  7.38246E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34079E+19 0.00127  1.04137E-01 0.00123 ];
U235_CAPT                 (idx, [1:   4]) = [  3.79574E+15 0.08353  2.94848E-05 0.08353 ];
SM149_CAPT                (idx, [1:   4]) = [  6.44454E+16 0.02012  5.00486E-04 0.02009 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000252 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15002E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000252 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5215695 5.22169E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3799775 3.80402E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 984782 9.85788E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000252 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36439E+20 3.7E-06  2.36439E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37760E+19 1.2E-07  9.37760E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28736E+20 0.00032  1.13795E+20 0.00023  1.49413E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22512E+20 0.00018  2.07571E+20 0.00012  1.49413E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46576E+20 0.00038  2.46576E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.40541E+22 0.00026  3.22307E+22 0.00024  1.82336E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43080E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46820E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10837E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29740E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29740E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.52257E-01 0.04683 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.04244E-01 0.01152 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.85933E-04 0.02019 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.35816E+03 0.02723 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01427E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.01563E-01 0.03309 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.22465E-01 0.03310 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52132E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.59194E-01 0.00041  1.06222E-01 0.00039  3.44165E-04 0.00857 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.59058E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.58929E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.59058E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06396E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03796E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03740E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31718E-02 0.00126 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31780E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50835E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51133E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38705E-03 0.00576  2.83128E-04 0.01917  8.29903E-04 0.01167  6.19030E-04 0.01238  1.32690E-03 0.00869  2.78478E-04 0.01954  4.96174E-05 0.04664 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42539E-01 0.01450  1.24761E-02 2.5E-05  3.23389E-02 8.6E-05  1.06051E-01 0.00055  2.97504E-01 0.00025  1.23555E+00 0.00022  5.20515E+00 0.04109 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26856E-03 0.00771  2.75874E-04 0.02862  7.97163E-04 0.01553  6.00705E-04 0.01965  1.27965E-03 0.01217  2.68239E-04 0.03097  4.69344E-05 0.06616 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38189E-01 0.02139  1.24760E-02 3.2E-05  3.23369E-02 0.00012  1.06104E-01 0.00081  2.97544E-01 0.00038  1.23534E+00 0.00031  6.71791E+00 0.02905 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.10251E-07 0.00138  9.09915E-07 0.00139  1.01692E-06 0.01708 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.73069E-07 0.00133  8.72746E-07 0.00134  9.75508E-07 0.01710 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23164E-03 0.00853  2.56375E-04 0.03387  8.00240E-04 0.01825  5.95262E-04 0.02178  1.27737E-03 0.01307  2.56413E-04 0.03240  4.59819E-05 0.07604 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.33486E-01 0.02388  1.24774E-02 3.1E-05  3.23437E-02 0.00016  1.05991E-01 0.00094  2.97384E-01 0.00044  1.23572E+00 0.00043  6.35722E+00 0.04344 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.08221E-07 0.01978  8.07861E-07 0.01978  9.18848E-07 0.05763 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.75289E-07 0.01977  7.74944E-07 0.01977  8.81283E-07 0.05762 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85793E-03 0.04547  2.36577E-04 0.14142  6.85782E-04 0.08068  5.07354E-04 0.09133  1.16101E-03 0.06410  2.19796E-04 0.13244  4.74141E-05 0.31477 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.38779E-01 0.10640  1.24794E-02 3.5E-06  3.23319E-02 0.00050  1.05846E-01 0.00304  2.96860E-01 0.00140  1.23353E+00 0.00125  6.26138E+00 0.12253 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87490E-03 0.04441  2.35798E-04 0.13861  7.05088E-04 0.07917  5.01989E-04 0.09104  1.16522E-03 0.06273  2.18702E-04 0.13149  4.80964E-05 0.30239 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.35699E-01 0.10193  1.24793E-02 4.7E-06  3.23321E-02 0.00051  1.05846E-01 0.00304  2.96910E-01 0.00140  1.23339E+00 0.00125  6.26138E+00 0.12253 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.56962E+03 0.04137 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.04801E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.67843E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23620E-03 0.00582 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.57724E+03 0.00587 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.96997E-09 0.00100 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.90905E-05 0.00574  6.91001E-05 0.00574  1.80168E-06 0.36955 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.76905E-05 0.02277  6.77035E-05 0.02276  1.56257E-06 0.50759 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.74744E-04 0.01957  2.74874E-04 0.01962  2.35131E-04 0.35090 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73427E+01 0.01130 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49009E+01 0.00029  4.05686E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 19 22:03:15 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01713E+00  1.00284E+00  1.00352E+00  9.96810E-01  1.00217E+00  1.00631E+00  9.98368E-01  9.96668E-01  9.45533E-01  1.00523E+00  1.00498E+00  9.99808E-01  1.00052E+00  1.00243E+00  1.00480E+00  1.00901E+00  1.00237E+00  1.00150E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.08923E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69108E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31471E-01 0.00015  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45754E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71776E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51703E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50725E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.52879E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.37408E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334180 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33380E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33380E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.54860E+03 ;
RUNNING_TIME              (idx, 1)        =  1.66524E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.58050E-01  1.34083E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.64828E+02  4.72215E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.12350E-01  1.88833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.14681E+01  3.20768E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.63361E+02  6.76033E+02 ];
CPU_USAGE                 (idx, 1)        = 9.29956 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98956E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.24439E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.51285E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67065E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.25692E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90607E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33222E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57506E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50564E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.82104E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.56366E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.02625E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.21497E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79367E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74214E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48783E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36011E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34462E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55773E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39870E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42534E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08763E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54496E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.27008E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.68952E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.46253E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.15981E-03 -2.49561E+26  2.15424E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.20082E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.60776E+18 0.00373  1.71500E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  9.18756E+19 0.00050  9.79973E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.57568E+16 0.02390  4.88005E-04 0.02389 ];
TH232_CAPT                (idx, [1:   4]) = [  9.62058E+19 0.00052  7.34714E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33708E+19 0.00136  1.02112E-01 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  1.66189E+16 0.03865  1.26900E-04 0.03863 ];
SM149_CAPT                (idx, [1:   4]) = [  1.02477E+17 0.01630  7.82646E-04 0.01629 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001410 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.17475E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001410 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5258319 5.26404E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3765228 3.76898E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 977863 9.78724E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001410 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.98606E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36431E+20 3.6E-06  2.36431E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.2E-07  9.37752E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.30893E+20 0.00028  1.15659E+20 0.00021  1.52343E+19 0.00153 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.24668E+20 0.00016  2.09434E+20 0.00011  1.52343E+19 0.00153 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48751E+20 0.00032  2.48751E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44181E+22 0.00022  3.25652E+22 0.00021  1.85289E+21 0.00144 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43467E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.49015E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12243E+22 0.00028 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29710E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29710E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.05981E-01 0.04834 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07064E-01 0.01274 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.19731E-04 0.02361 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.92901E+03 0.02775 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02130E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.72472E-01 0.03488 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.96981E-01 0.03488 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52125E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.50032E-01 0.00044  1.05238E-01 0.00043  3.46607E-04 0.00915 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.50567E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.50501E-01 0.00031 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.50567E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05370E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04108E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04051E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31308E-02 0.00127 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31368E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52297E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52101E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44548E-03 0.00564  2.79080E-04 0.02140  8.36462E-04 0.01055  6.39099E-04 0.01348  1.35425E-03 0.00833  2.82858E-04 0.02030  5.37355E-05 0.04364 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.49207E-01 0.01466  1.24760E-02 2.5E-05  3.23396E-02 9.2E-05  1.06095E-01 0.00055  2.97582E-01 0.00027  1.23560E+00 0.00025  5.37854E+00 0.03809 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26059E-03 0.00821  2.64753E-04 0.02965  7.88804E-04 0.01643  6.05412E-04 0.01978  1.28570E-03 0.01262  2.63456E-04 0.02927  5.24615E-05 0.06121 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.53042E-01 0.02260  1.24758E-02 3.6E-05  3.23412E-02 0.00013  1.06075E-01 0.00078  2.97410E-01 0.00036  1.23576E+00 0.00032  6.47810E+00 0.02900 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.20411E-07 0.00123  9.20063E-07 0.00123  1.02786E-06 0.01542 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.74358E-07 0.00111  8.74028E-07 0.00111  9.76379E-07 0.01540 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28224E-03 0.00941  2.68340E-04 0.03530  7.97377E-04 0.01666  6.17740E-04 0.02177  1.27065E-03 0.01376  2.76860E-04 0.02965  5.12719E-05 0.07209 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.52720E-01 0.02610  1.24759E-02 4.3E-05  3.23415E-02 0.00016  1.06006E-01 0.00093  2.97719E-01 0.00049  1.23517E+00 0.00042  6.61576E+00 0.04019 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.09651E-07 0.01955  8.09333E-07 0.01956  8.86329E-07 0.05543 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.69250E-07 0.01956  7.68948E-07 0.01956  8.41894E-07 0.05537 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.17176E-03 0.04335  2.72793E-04 0.14695  7.99306E-04 0.07722  6.53484E-04 0.08985  1.15688E-03 0.06326  2.40319E-04 0.13206  4.89772E-05 0.31302 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.65233E-01 0.10936  1.24740E-02 0.00017  3.23286E-02 0.00047  1.05732E-01 0.00270  2.97577E-01 0.00169  1.23464E+00 0.00118  7.68898E+00 0.09730 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.16247E-03 0.04284  2.73821E-04 0.14488  7.95994E-04 0.07637  6.32639E-04 0.08880  1.16988E-03 0.06164  2.44581E-04 0.12804  4.55532E-05 0.29852 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.58206E-01 0.09789  1.24739E-02 0.00017  3.23291E-02 0.00047  1.05686E-01 0.00263  2.97527E-01 0.00166  1.23452E+00 0.00118  7.58860E+00 0.09911 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.93635E+03 0.03910 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.13170E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.67493E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39286E-03 0.00667 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.71495E+03 0.00656 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89815E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.87072E-05 0.00587  6.87092E-05 0.00588  6.40605E-07 0.60325 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.08154E-05 0.02370  6.07251E-05 0.02355  1.15747E-06 0.70689 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.11121E-04 0.02253  2.11545E-04 0.02263  9.21499E-05 0.57701 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70095E+01 0.01255 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50725E+01 0.00024  4.06903E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 19 22:52:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01481E+00  1.00906E+00  1.00832E+00  1.00509E+00  9.99440E-01  9.97422E-01  9.98279E-01  9.97966E-01  9.31262E-01  1.00459E+00  1.00568E+00  1.00384E+00  1.00615E+00  1.00404E+00  9.97755E-01  1.00666E+00  1.00422E+00  1.00542E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.08305E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69170E-01 9.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31274E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45541E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71181E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53174E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52196E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.56472E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.37478E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334030 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33393E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33393E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.00747E+03 ;
RUNNING_TIME              (idx, 1)        =  2.15467E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.91350E-01  1.33300E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.13604E+02  4.87765E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.30617E-01  1.82667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.47254E+01  3.25730E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.12251E+02  6.85981E+02 ];
CPU_USAGE                 (idx, 1)        = 9.31684 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98960E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26207E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.54072E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66522E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.65906E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93018E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34738E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57853E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49842E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00136E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.68445E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.31099E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.32344E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69669E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85199E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93570E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32299E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31611E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51853E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35042E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81372E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.08571E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51772E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.55842E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.73884E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.56529E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64039E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.35524E-03 -5.06787E+26  2.15681E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.43976E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.61645E+18 0.00400  1.72405E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  9.15191E+19 0.00050  9.76170E-01 8.1E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.75476E+17 0.01309  1.87190E-03 0.01310 ];
TH232_CAPT                (idx, [1:   4]) = [  9.78898E+19 0.00052  7.30359E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33702E+19 0.00132  9.97574E-02 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  5.95848E+16 0.02097  4.44621E-04 0.02097 ];
SM149_CAPT                (idx, [1:   4]) = [  1.39988E+17 0.01300  1.04465E-03 0.01305 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001804 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16138E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001804 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5309651 5.31495E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3714151 3.71779E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 978002 9.78873E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001804 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.45058E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36415E+20 3.8E-06  2.36415E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34006E+20 0.00028  1.18589E+20 0.00021  1.54166E+19 0.00170 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27779E+20 0.00017  2.12362E+20 0.00012  1.54166E+19 0.00170 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.52176E+20 0.00036  2.52176E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49737E+22 0.00025  3.31005E+22 0.00022  1.87322E+21 0.00163 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46858E+19 0.00136 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.52465E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14158E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29353E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29353E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.05655E-01 0.04580 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.35970E-01 0.01120 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.97124E-04 0.02308 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.57829E+03 0.02873 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02116E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 7.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.78913E-01 0.03339 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.02723E-01 0.03339 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52115E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99680E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.37310E-01 0.00042  1.03808E-01 0.00040  3.36042E-04 0.00946 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.37516E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.37536E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.37516E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03925E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04255E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04367E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31125E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30956E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54044E-01 0.00162 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54263E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.48260E-03 0.00591  2.88968E-04 0.02042  8.47162E-04 0.01155  6.36235E-04 0.01388  1.36710E-03 0.00861  2.89472E-04 0.01846  5.36615E-05 0.04490 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47194E-01 0.01473  1.24753E-02 2.7E-05  3.23344E-02 9.3E-05  1.06181E-01 0.00058  2.97526E-01 0.00026  1.23606E+00 0.00028  5.45156E+00 0.03752 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28504E-03 0.00772  2.67886E-04 0.02894  8.01571E-04 0.01620  6.04302E-04 0.01940  1.28492E-03 0.01219  2.75548E-04 0.02739  5.08191E-05 0.06921 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.43805E-01 0.02211  1.24755E-02 3.9E-05  3.23328E-02 0.00014  1.06157E-01 0.00082  2.97534E-01 0.00038  1.23618E+00 0.00039  6.51435E+00 0.02909 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.36185E-07 0.00134  9.35819E-07 0.00135  1.04918E-06 0.01899 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.77458E-07 0.00130  8.77115E-07 0.00130  9.83486E-07 0.01903 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22656E-03 0.00947  2.70500E-04 0.03320  7.77599E-04 0.01820  6.01514E-04 0.02111  1.25407E-03 0.01510  2.70337E-04 0.03307  5.25401E-05 0.07315 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.49700E-01 0.02502  1.24765E-02 3.7E-05  3.23315E-02 0.00016  1.06107E-01 0.00100  2.97478E-01 0.00044  1.23620E+00 0.00047  6.43702E+00 0.04085 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.30289E-07 0.01970  8.29965E-07 0.01970  8.99143E-07 0.05528 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.78128E-07 0.01969  7.77825E-07 0.01970  8.42936E-07 0.05532 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.07817E-03 0.04571  2.51281E-04 0.13487  7.36325E-04 0.08755  5.51085E-04 0.09887  1.29282E-03 0.06489  2.06282E-04 0.15634  4.03768E-05 0.29676 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.03524E-01 0.09640  1.24796E-02 1.3E-05  3.23495E-02 0.00057  1.05594E-01 0.00264  2.96655E-01 0.00148  1.23756E+00 0.00195  5.45663E+00 0.15206 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03544E-03 0.04536  2.40233E-04 0.13108  7.26826E-04 0.08673  5.40849E-04 0.09786  1.28192E-03 0.06435  2.07278E-04 0.15184  3.83356E-05 0.29057 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.98513E-01 0.08451  1.24796E-02 1.2E-05  3.23496E-02 0.00057  1.05605E-01 0.00264  2.96603E-01 0.00145  1.23756E+00 0.00195  5.45663E+00 0.15206 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.73746E+03 0.04155 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.29055E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.70772E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26163E-03 0.00637 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.51127E+03 0.00642 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.88887E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79795E-05 0.00768  6.80308E-05 0.00773  1.96608E-06 0.31805 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.77547E-05 0.02546  5.77855E-05 0.02540  1.88911E-06 0.43628 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.90055E-04 0.02219  1.89718E-04 0.02238  2.84084E-04 0.31293 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72341E+01 0.01182 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52196E+01 0.00025  4.07893E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 19 23:41:54 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01299E+00  1.00823E+00  9.95343E-01  1.01258E+00  1.00130E+00  1.00367E+00  1.00050E+00  1.00517E+00  9.33247E-01  9.96759E-01  9.99797E-01  9.99885E-01  1.00538E+00  1.01322E+00  9.96250E-01  1.00386E+00  1.00347E+00  1.00835E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.08084E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69192E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31124E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45388E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70975E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54217E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53236E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59021E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.37707E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334413 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33403E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33403E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.46139E+03 ;
RUNNING_TIME              (idx, 1)        =  2.65183E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.25733E-01  1.34383E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.63151E+02  4.95473E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.48767E-01  1.81500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.93756E+01  4.65008E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.60575E+02  7.00936E+02 ];
CPU_USAGE                 (idx, 1)        = 9.28184 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98862E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.22269E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.61313E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67547E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.14742E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96826E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37263E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61246E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50577E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10064E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79678E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.73791E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.49742E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25319E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94673E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35048E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33427E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32435E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52976E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02114E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17174E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.12830E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55248E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.44889E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.84736E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.65364E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96058E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.53190E-03 -7.59975E+26  2.15934E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.66048E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.63944E+18 0.00372  1.74910E-02 0.00365 ];
U233_FISS                 (idx, [1:   4]) = [  9.10780E+19 0.00051  9.71743E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.65512E+17 0.00828  3.89974E-03 0.00826 ];
TH232_CAPT                (idx, [1:   4]) = [  9.93469E+19 0.00051  7.27112E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33070E+19 0.00146  9.73914E-02 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  1.30581E+17 0.01362  9.55804E-04 0.01364 ];
SM149_CAPT                (idx, [1:   4]) = [  1.58106E+17 0.01247  1.15724E-03 0.01249 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002094 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16906E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002094 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5350300 5.35563E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3670405 3.67382E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981389 9.82239E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002094 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.92435E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36399E+20 3.6E-06  2.36399E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37698E+19 1.1E-07  9.37698E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36566E+20 0.00030  1.20912E+20 0.00022  1.56538E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30335E+20 0.00018  2.14682E+20 0.00013  1.56538E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.55121E+20 0.00034  2.55121E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54449E+22 0.00025  3.35499E+22 0.00024  1.89495E+21 0.00169 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50596E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.55395E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15758E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28857E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28857E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.13323E-01 0.04599 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.05251E-01 0.01394 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93659E-04 0.02335 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.04087E+03 0.02701 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01779E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.84104E-01 0.03220 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.07099E-01 0.03220 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52106E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.26098E-01 0.00042  1.02568E-01 0.00042  3.40252E-04 0.00855 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.26690E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.26645E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.26690E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02764E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04221E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04284E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31166E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31064E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56631E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.56719E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.56755E-03 0.00525  2.87192E-04 0.01969  8.60464E-04 0.01144  6.46810E-04 0.01359  1.40858E-03 0.00833  3.04202E-04 0.01846  6.02988E-05 0.04191 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.61413E-01 0.01505  1.24752E-02 3.0E-05  3.23220E-02 9.3E-05  1.06190E-01 0.00059  2.97954E-01 0.00029  1.23664E+00 0.00030  5.44799E+00 0.03495 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26737E-03 0.00779  2.56602E-04 0.02953  7.84846E-04 0.01672  5.91507E-04 0.01994  1.29746E-03 0.01365  2.81679E-04 0.02889  5.52744E-05 0.05523 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.63689E-01 0.01990  1.24752E-02 3.9E-05  3.23218E-02 0.00013  1.06246E-01 0.00083  2.97913E-01 0.00039  1.23669E+00 0.00041  6.46666E+00 0.02767 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.50462E-07 0.00130  9.50139E-07 0.00129  1.05062E-06 0.01636 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.80172E-07 0.00122  8.79873E-07 0.00122  9.72731E-07 0.01631 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30322E-03 0.00876  2.58369E-04 0.03166  7.90756E-04 0.01891  6.03463E-04 0.02197  1.31387E-03 0.01357  2.77504E-04 0.03268  5.92597E-05 0.06918 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.65134E-01 0.02510  1.24755E-02 5.1E-05  3.23247E-02 0.00016  1.06043E-01 0.00097  2.97784E-01 0.00044  1.23619E+00 0.00057  6.45731E+00 0.03850 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.43603E-07 0.01961  8.43157E-07 0.01961  9.61074E-07 0.05887 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.81316E-07 0.01961  7.80903E-07 0.01961  8.90271E-07 0.05887 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00306E-03 0.04215  2.42177E-04 0.12803  7.13355E-04 0.07636  5.91173E-04 0.08688  1.16850E-03 0.06581  2.63897E-04 0.15721  2.39598E-05 0.36159 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.09995E-01 0.10077  1.24773E-02 0.00010  3.23512E-02 0.00063  1.05956E-01 0.00297  2.96743E-01 0.00140  1.23930E+00 0.00239  6.06852E+00 0.14198 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.02437E-03 0.04144  2.45290E-04 0.12770  7.22288E-04 0.07591  6.01219E-04 0.08512  1.16978E-03 0.06347  2.62672E-04 0.15537  2.31233E-05 0.35412 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.07320E-01 0.09605  1.24774E-02 0.00010  3.23529E-02 0.00063  1.05921E-01 0.00287  2.96830E-01 0.00141  1.23931E+00 0.00239  6.11433E+00 0.14043 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.58174E+03 0.03783 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.42462E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.72762E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32306E-03 0.00620 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.52647E+03 0.00623 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91386E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70238E-05 0.00653  6.70351E-05 0.00655  4.17679E-07 0.74524 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.99927E-05 0.02590  5.99997E-05 0.02589  3.18415E-07 0.91490 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.86760E-04 0.02226  1.87228E-04 0.02226  5.85153E-05 0.70606 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73781E+01 0.01108 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53236E+01 0.00026  4.08465E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 00:32:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01685E+00  1.01063E+00  1.01033E+00  9.98482E-01  9.96993E-01  1.00324E+00  1.00319E+00  1.00473E+00  9.35849E-01  1.00830E+00  1.00404E+00  1.00049E+00  1.00103E+00  1.00597E+00  9.92539E-01  1.00351E+00  1.00346E+00  1.00036E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.07808E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69219E-01 9.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30682E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44944E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70868E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55501E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54517E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.63140E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38098E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334335 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33438E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33438E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.91850E+03 ;
RUNNING_TIME              (idx, 1)        =  3.15846E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.66083E-01  1.40350E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.13636E+02  5.04849E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.67817E-01  1.90500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.47345E+01  5.35888E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.10531E+02  7.08184E+02 ];
CPU_USAGE                 (idx, 1)        = 9.24024 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98978E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.17567E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.64366E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67408E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.21923E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99702E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38990E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61381E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50225E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27568E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.93161E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.77238E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.65771E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93736E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06491E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06754E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30724E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30258E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50041E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.83741E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78418E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13532E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.56489E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.18544E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.89704E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77032E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60097E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.97521E-03 -1.28571E+27  2.16460E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.96091E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.64509E+18 0.00384  1.75499E-02 0.00380 ];
U233_FISS                 (idx, [1:   4]) = [  9.01763E+19 0.00051  9.62022E-01 9.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  9.00076E+17 0.00494  9.60169E-03 0.00488 ];
PU239_FISS                (idx, [1:   4]) = [  2.57067E+13 1.00000  2.75778E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01018E+20 0.00049  7.21340E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31550E+19 0.00143  9.39373E-02 0.00140 ];
U235_CAPT                 (idx, [1:   4]) = [  3.16875E+17 0.00874  2.26277E-03 0.00875 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74443E+17 0.01234  1.24562E-03 0.01232 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003126 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14176E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003126 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5402164 5.40684E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3616134 3.61903E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 984828 9.85547E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003126 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.92435E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36363E+20 3.8E-06  2.36363E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37618E+19 1.2E-07  9.37618E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40085E+20 0.00029  1.24148E+20 0.00021  1.59377E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33847E+20 0.00017  2.17909E+20 0.00012  1.59377E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59011E+20 0.00034  2.59011E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.60877E+22 0.00025  3.41649E+22 0.00022  1.92284E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.55275E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59375E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17853E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27843E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27843E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.96979E-01 0.04645 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.36156E-01 0.01189 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.83973E-04 0.02476 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.55507E+03 0.02811 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01448E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.49099E-01 0.03428 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.75328E-01 0.03428 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52089E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99703E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.12290E-01 0.00043  1.01035E-01 0.00042  3.34880E-04 0.00913 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12341E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.12595E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12341E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01206E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03759E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03523E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31772E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32064E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.61324E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.61638E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.64120E-03 0.00576  2.96808E-04 0.01889  8.77885E-04 0.01093  6.77059E-04 0.01283  1.41264E-03 0.00950  3.17852E-04 0.01900  5.89535E-05 0.04258 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.55105E-01 0.01464  1.24757E-02 2.6E-05  3.23172E-02 0.00010  1.06330E-01 0.00057  2.97991E-01 0.00028  1.23531E+00 0.00337  5.31075E+00 0.03652 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29454E-03 0.00827  2.71763E-04 0.02752  7.93985E-04 0.01514  6.22483E-04 0.01803  1.27435E-03 0.01409  2.80115E-04 0.02863  5.18373E-05 0.06253 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46516E-01 0.01941  1.24753E-02 3.9E-05  3.23148E-02 0.00015  1.06380E-01 0.00084  2.98035E-01 0.00039  1.23926E+00 0.00058  6.35111E+00 0.02847 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.64551E-07 0.00109  9.64125E-07 0.00109  1.09024E-06 0.01622 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.79903E-07 0.00100  8.79514E-07 0.00100  9.94478E-07 0.01620 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30559E-03 0.00915  2.64811E-04 0.03086  7.95469E-04 0.01884  6.25861E-04 0.02118  1.26872E-03 0.01542  2.98394E-04 0.03156  5.23356E-05 0.07619 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51752E-01 0.02592  1.24757E-02 4.6E-05  3.23099E-02 0.00017  1.06272E-01 0.00104  2.98084E-01 0.00048  1.24007E+00 0.00071  6.00589E+00 0.04323 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.54927E-07 0.01963  8.54391E-07 0.01963  9.94042E-07 0.04720 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.80012E-07 0.01963  7.79524E-07 0.01963  9.06378E-07 0.04714 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.96843E-03 0.04412  2.44316E-04 0.12273  8.29032E-04 0.08527  5.02127E-04 0.09275  1.07848E-03 0.06832  2.71437E-04 0.12339  4.30307E-05 0.27735 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.03040E-01 0.10391  1.24749E-02 0.00016  3.23015E-02 0.00059  1.06057E-01 0.00308  2.97952E-01 0.00173  1.23503E+00 0.00154  6.49214E+00 0.12125 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97475E-03 0.04308  2.42781E-04 0.12175  8.44153E-04 0.08157  5.00689E-04 0.08857  1.08334E-03 0.06592  2.61385E-04 0.12045  4.23958E-05 0.29104 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.96869E-01 0.10597  1.24749E-02 0.00016  3.23023E-02 0.00060  1.06140E-01 0.00318  2.97990E-01 0.00174  1.23518E+00 0.00154  6.49214E+00 0.12125 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.48302E+03 0.03960 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.58215E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.74118E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32077E-03 0.00569 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.46544E+03 0.00562 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91444E-09 0.00070 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68725E-05 0.00760  6.68453E-05 0.00762  1.60660E-06 0.42458 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.75362E-05 0.02890  5.73708E-05 0.02897  2.20597E-06 0.57087 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.76056E-04 0.02361  1.76101E-04 0.02357  1.58538E-04 0.40681 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71440E+01 0.01144 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54517E+01 0.00028  4.08646E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 01:23:54 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02207E+00  1.00378E+00  1.00761E+00  1.00455E+00  9.96961E-01  1.01192E+00  1.00342E+00  1.01074E+00  9.29483E-01  9.93114E-01  9.94158E-01  9.99779E-01  9.97946E-01  1.00269E+00  9.99828E-01  1.01355E+00  1.00502E+00  1.00340E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.08146E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69185E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30349E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44630E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71087E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55626E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54639E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.64574E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38495E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333394 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33329E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33329E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.38227E+03 ;
RUNNING_TIME              (idx, 1)        =  3.67174E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.10737E+00  1.41283E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.64780E+02  5.11440E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.87283E-01  1.94667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.00830E+01  5.34845E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.61868E+02  7.16388E+02 ];
CPU_USAGE                 (idx, 1)        = 9.21164 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98983E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.14336E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.73115E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68993E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.21535E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02751E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40550E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67004E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51582E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.77240E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.23711E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.32167E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.91886E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23737E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.34198E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32941E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30928E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30059E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49958E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.56468E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27964E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.17926E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.68276E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.89981E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99296E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.90084E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.21176E-02 -2.60740E+27  2.17782E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02990E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.65148E+18 0.00437  1.76151E-02 0.00436 ];
U233_FISS                 (idx, [1:   4]) = [  8.78279E+19 0.00056  9.36766E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.47714E+18 0.00351  2.64211E-02 0.00347 ];
PU239_FISS                (idx, [1:   4]) = [  1.31584E+15 0.14106  1.40036E-05 0.14115 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01804E+20 0.00051  7.07478E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28107E+19 0.00141  8.90273E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  8.76478E+17 0.00561  6.09120E-03 0.00562 ];
U238_CAPT                 (idx, [1:   4]) = [  2.60796E+13 1.00000  1.78904E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  6.04045E+14 0.20995  4.19256E-06 0.21019 ];
PU240_CAPT                (idx, [1:   4]) = [  5.28577E+13 0.70595  3.63494E-07 0.70596 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80141E+17 0.01148  1.25185E-03 0.01148 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 9999862 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14069E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 9999862 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5457478 5.46392E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3555937 3.56000E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 986447 9.87480E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 9999862 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.27243E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36263E+20 4.3E-06  2.36263E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37386E+19 1.3E-07  9.37386E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43902E+20 0.00031  1.27671E+20 0.00022  1.62314E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37641E+20 0.00019  2.21409E+20 0.00013  1.62314E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.63361E+20 0.00036  2.63361E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67209E+22 0.00025  3.47706E+22 0.00023  1.95032E+21 0.00164 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60073E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.63648E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19864E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.25218E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.25218E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.94985E-01 0.04640 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31732E-01 0.01322 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94631E-04 0.02507 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.40541E+03 0.02774 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01255E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.49997E-01 0.03310 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.75831E-01 0.03310 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52045E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99753E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.97222E-01 0.00045  9.93737E-02 0.00044  3.24054E-04 0.00931 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.97167E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.97141E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.97167E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95474E-01 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00517E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00493E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36122E-02 0.00149 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36129E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.71624E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.71878E-01 0.00053 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.75749E-03 0.00571  2.94199E-04 0.01961  9.19685E-04 0.01141  6.91932E-04 0.01253  1.46151E-03 0.00881  3.24928E-04 0.01797  6.52347E-05 0.04438 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.65386E-01 0.01446  1.24757E-02 3.6E-05  3.22955E-02 0.00011  1.06291E-01 0.00055  2.98953E-01 0.00031  1.24548E+00 0.00056  5.48595E+00 0.03481 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.32427E-03 0.00805  2.63026E-04 0.02892  8.15576E-04 0.01560  6.24120E-04 0.01848  1.27587E-03 0.01236  2.87159E-04 0.02831  5.85281E-05 0.06130 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.70296E-01 0.02281  1.24758E-02 4.4E-05  3.22857E-02 0.00015  1.06325E-01 0.00082  2.99002E-01 0.00042  1.24646E+00 0.00083  6.56346E+00 0.02577 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.71340E-07 0.00126  9.70952E-07 0.00126  1.07863E-06 0.02081 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.71436E-07 0.00111  8.71088E-07 0.00111  9.67910E-07 0.02088 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.25780E-03 0.00938  2.53968E-04 0.03442  7.97255E-04 0.01897  6.07916E-04 0.02075  1.26401E-03 0.01558  2.73564E-04 0.03010  6.10867E-05 0.07528 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.70485E-01 0.02672  1.24757E-02 5.3E-05  3.22951E-02 0.00021  1.06374E-01 0.00102  2.99071E-01 0.00052  1.24377E+00 0.00091  6.49105E+00 0.03746 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.63462E-07 0.01964  8.63050E-07 0.01965  9.25306E-07 0.05192 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.74794E-07 0.01964  7.74426E-07 0.01965  8.30289E-07 0.05195 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.99530E-03 0.04056  2.68954E-04 0.12860  7.14186E-04 0.07834  5.25971E-04 0.09164  1.18569E-03 0.06634  2.72160E-04 0.13351  2.83351E-05 0.33128 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.22005E-01 0.07677  1.24740E-02 0.00018  3.22820E-02 0.00054  1.06887E-01 0.00366  2.98847E-01 0.00189  1.24108E+00 0.00218  5.74621E+00 0.13135 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98784E-03 0.03980  2.68918E-04 0.12851  7.13045E-04 0.07641  5.14428E-04 0.09012  1.19374E-03 0.06525  2.69977E-04 0.13118  2.77324E-05 0.29827 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.24841E-01 0.07829  1.24741E-02 0.00018  3.22831E-02 0.00054  1.06898E-01 0.00366  2.98872E-01 0.00189  1.24074E+00 0.00212  5.75117E+00 0.13147 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.49514E+03 0.03601 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.65225E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.65967E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35002E-03 0.00589 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.47190E+03 0.00604 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93219E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69142E-05 0.00820  6.68813E-05 0.00820  2.06804E-06 0.36556 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.66399E-05 0.02554  5.65994E-05 0.02557  1.89394E-06 0.41059 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85199E-04 0.02427  1.85096E-04 0.02430  2.17202E-04 0.35174 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69400E+01 0.01152 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54639E+01 0.00028  4.06226E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 02:15:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01452E+00  1.00138E+00  1.00656E+00  1.01076E+00  9.98551E-01  1.00584E+00  9.98070E-01  1.00770E+00  9.32652E-01  9.93289E-01  1.00780E+00  1.00772E+00  1.00476E+00  1.00384E+00  9.98776E-01  1.00173E+00  9.98031E-01  1.00802E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.09081E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69092E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30316E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44630E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70548E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53938E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52953E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61366E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38625E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333731 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33362E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33362E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.85098E+03 ;
RUNNING_TIME              (idx, 1)        =  4.18976E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.25147E+00  1.44100E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.16397E+02  5.16169E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.05500E-01  1.82167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.54069E+01  5.32380E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.13692E+02  7.21530E+02 ];
CPU_USAGE                 (idx, 1)        = 9.19141 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98983E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.12054E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.89865E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73531E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.81166E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03536E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40534E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82892E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56054E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.21850E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78677E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.13681E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.28912E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04675E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85097E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38258E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38325E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35127E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57226E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.85317E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30650E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28089E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.94236E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.48400E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14230E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.94306E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64364E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.28444E-02 -4.91554E+27  2.20090E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03902E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.61190E+18 0.00403  1.72083E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  8.43138E+19 0.00056  9.00191E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.89456E+18 0.00231  5.22571E-02 0.00223 ];
U238_FISS                 (idx, [1:   4]) = [  2.60828E+13 1.00000  2.78033E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  2.55020E+16 0.03364  2.72289E-04 0.03368 ];
PU240_FISS                (idx, [1:   4]) = [  9.49546E+14 0.17053  1.01244E-05 0.17058 ];
PU241_FISS                (idx, [1:   4]) = [  6.61110E+14 0.19182  7.06693E-06 0.19182 ];
TH232_CAPT                (idx, [1:   4]) = [  9.95441E+19 0.00056  6.85065E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22992E+19 0.00149  8.46450E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  1.73051E+18 0.00404  1.19099E-02 0.00405 ];
U238_CAPT                 (idx, [1:   4]) = [  1.05195E+14 0.49755  7.22339E-07 0.49750 ];
PU239_CAPT                (idx, [1:   4]) = [  1.19968E+16 0.04677  8.25478E-05 0.04674 ];
PU240_CAPT                (idx, [1:   4]) = [  2.25416E+15 0.10725  1.54895E-05 0.10719 ];
PU241_CAPT                (idx, [1:   4]) = [  3.17137E+14 0.32877  2.18884E-06 0.32906 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84973E+17 0.01174  1.27315E-03 0.01177 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000847 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11438E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000847 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5482121 5.48804E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3534108 3.53754E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 984618 9.85557E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000847 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.32717E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36141E+20 4.0E-06  2.36141E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36992E+19 1.1E-07  9.36992E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45312E+20 0.00030  1.29024E+20 0.00021  1.62880E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39012E+20 0.00018  2.22724E+20 0.00012  1.62880E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64769E+20 0.00035  2.64769E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68335E+22 0.00025  3.48728E+22 0.00023  1.96077E+21 0.00169 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60953E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65107E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20055E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.18585E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.18585E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88308E-01 0.04969 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.28556E-01 0.01228 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94783E-04 0.02638 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.42962E+03 0.02966 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01447E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.15537E-01 0.03577 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.45053E-01 0.03577 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52020E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99837E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.91559E-01 0.00049  9.87231E-02 0.00047  3.39829E-04 0.00880 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.91745E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  8.91910E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.91745E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89222E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94541E+00 0.00023 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94377E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44510E-02 0.00156 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44715E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86035E-01 0.00170 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.85704E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94907E-03 0.00554  2.96887E-04 0.01987  9.30828E-04 0.01096  7.15113E-04 0.01396  1.58160E-03 0.00842  3.46339E-04 0.01791  7.83037E-05 0.03686 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.93269E-01 0.01491  1.24765E-02 4.0E-05  3.22665E-02 0.00012  1.06543E-01 0.00051  3.00121E-01 0.00034  1.25376E+00 0.00063  6.15551E+00 0.02728 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46524E-03 0.00765  2.65039E-04 0.02800  8.12420E-04 0.01620  6.35213E-04 0.01933  1.37591E-03 0.01234  3.02495E-04 0.02565  7.41687E-05 0.05579 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02320E-01 0.02310  1.24766E-02 7.2E-05  3.22625E-02 0.00017  1.06439E-01 0.00070  2.99999E-01 0.00044  1.25393E+00 0.00089  6.67918E+00 0.02306 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.52812E-07 0.00132  9.52431E-07 0.00132  1.06366E-06 0.01664 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.49425E-07 0.00121  8.49084E-07 0.00121  9.48390E-07 0.01667 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43686E-03 0.00887  2.64384E-04 0.03363  8.01523E-04 0.01855  6.11826E-04 0.02309  1.39306E-03 0.01399  2.97189E-04 0.03059  6.88765E-05 0.06255 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.93726E-01 0.02543  1.24758E-02 5.5E-05  3.22604E-02 0.00021  1.06533E-01 0.00103  2.99866E-01 0.00049  1.25585E+00 0.00123  6.69536E+00 0.03181 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.43577E-07 0.01962  8.43384E-07 0.01962  9.29306E-07 0.06781 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.52242E-07 0.01962  7.52071E-07 0.01963  8.28497E-07 0.06783 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89953E-03 0.04291  2.12473E-04 0.15071  6.56297E-04 0.08610  5.19038E-04 0.09556  1.18636E-03 0.06160  2.74710E-04 0.14167  5.06527E-05 0.34218 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.88913E-01 0.12440  1.24726E-02 0.00022  3.22701E-02 0.00078  1.07044E-01 0.00371  3.00485E-01 0.00204  1.25034E+00 0.00315  7.30764E+00 0.08912 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91409E-03 0.04261  2.21047E-04 0.14758  6.71736E-04 0.08708  5.18546E-04 0.09580  1.17514E-03 0.06088  2.71004E-04 0.13925  5.66170E-05 0.32816 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.90788E-01 0.12184  1.24727E-02 0.00022  3.22712E-02 0.00079  1.07110E-01 0.00377  3.00506E-01 0.00203  1.25014E+00 0.00314  7.30764E+00 0.08912 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.45610E+03 0.03886 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.45789E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.43167E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35092E-03 0.00542 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.54364E+03 0.00548 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90323E-09 0.00086 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69154E-05 0.00893  6.68853E-05 0.00892  1.06550E-06 0.50820 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.56210E-05 0.02456  5.56471E-05 0.02457  7.76803E-07 0.53968 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.87089E-04 0.02578  1.87429E-04 0.02582  9.88591E-05 0.50061 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64211E+01 0.01174 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52953E+01 0.00027  4.00978E+01 0.00040 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 03:08:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01581E+00  9.99339E-01  1.00139E+00  1.00509E+00  9.99143E-01  1.00824E+00  9.95772E-01  1.00442E+00  9.31486E-01  1.00650E+00  1.00686E+00  1.00367E+00  1.00596E+00  1.00731E+00  9.95669E-01  1.00654E+00  9.99099E-01  1.00771E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.09570E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69043E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30397E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44727E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70795E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53181E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52201E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59568E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38635E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333785 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33381E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33381E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.32669E+03 ;
RUNNING_TIME              (idx, 1)        =  4.71560E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.39843E+00  1.46967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.68787E+02  5.23899E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.24250E-01  1.87500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.08151E+01  5.40822E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.66194E+02  7.25107E+02 ];
CPU_USAGE                 (idx, 1)        = 9.17527 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98993E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.10233E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.92342E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74178E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.24045E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02872E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40031E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86016E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56736E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.20874E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99940E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.03381E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.47849E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13539E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04376E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81876E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38584E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35193E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57383E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90027E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73647E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29115E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99319E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.91854E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.15504E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.94421E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30374E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.80631E-02 -6.03847E+27  2.21213E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03575E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.60296E+18 0.00425  1.71021E-02 0.00415 ];
U233_FISS                 (idx, [1:   4]) = [  8.33028E+19 0.00057  8.88841E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  5.57374E+18 0.00209  5.94731E-02 0.00206 ];
U238_FISS                 (idx, [1:   4]) = [  2.68697E+13 1.00000  2.90918E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  5.98489E+16 0.02081  6.38599E-04 0.02083 ];
PU240_FISS                (idx, [1:   4]) = [  1.93235E+15 0.11263  2.05999E-05 0.11261 ];
PU241_FISS                (idx, [1:   4]) = [  2.91595E+15 0.09806  3.11648E-05 0.09822 ];
TH232_CAPT                (idx, [1:   4]) = [  9.82940E+19 0.00053  6.75961E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21713E+19 0.00142  8.37026E-02 0.00140 ];
U235_CAPT                 (idx, [1:   4]) = [  1.97959E+18 0.00364  1.36137E-02 0.00363 ];
U238_CAPT                 (idx, [1:   4]) = [  2.66957E+14 0.31146  1.83754E-06 0.31145 ];
PU239_CAPT                (idx, [1:   4]) = [  2.69899E+16 0.03233  1.85587E-04 0.03231 ];
PU240_CAPT                (idx, [1:   4]) = [  5.92208E+15 0.07161  4.07730E-05 0.07160 ];
PU241_CAPT                (idx, [1:   4]) = [  7.68773E+14 0.18361  5.28553E-06 0.18363 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84398E+17 0.01282  1.26814E-03 0.01282 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001418 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11941E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001418 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5485732 5.49133E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3535913 3.53922E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 979773 9.80645E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001418 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.58794E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36139E+20 4.1E-06  2.36139E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36848E+19 1.0E-07  9.36848E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45365E+20 0.00030  1.29023E+20 0.00020  1.63413E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39049E+20 0.00018  2.22708E+20 0.00012  1.63413E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64807E+20 0.00034  2.64807E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67988E+22 0.00024  3.48311E+22 0.00022  1.96766E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59689E+19 0.00133 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65018E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19873E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.15170E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.15170E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.92191E-01 0.04698 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.50812E-01 0.01093 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93926E-04 0.02394 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.51493E+03 0.02771 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01938E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.64804E-01 0.03131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.89781E-01 0.03131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52057E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99867E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.91976E-01 0.00046  9.87796E-02 0.00044  3.41454E-04 0.00899 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92028E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.91768E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92028E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89050E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92149E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92084E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47998E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.48068E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90926E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90686E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93913E-03 0.00530  2.85313E-04 0.01980  9.30243E-04 0.01160  6.99567E-04 0.01283  1.58969E-03 0.00755  3.56457E-04 0.01737  7.78556E-05 0.03812 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.94231E-01 0.01361  1.24760E-02 3.3E-05  3.22516E-02 0.00013  1.06713E-01 0.00059  3.00501E-01 0.00028  1.25665E+00 0.00083  5.86760E+00 0.02938 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45833E-03 0.00797  2.47571E-04 0.02793  8.23781E-04 0.01613  6.11386E-04 0.01772  1.39605E-03 0.01159  3.11188E-04 0.02527  6.83502E-05 0.05699 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.95723E-01 0.02091  1.24762E-02 3.9E-05  3.22520E-02 0.00022  1.06598E-01 0.00080  3.00567E-01 0.00043  1.25632E+00 0.00102  6.68624E+00 0.02300 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.41121E-07 0.00134  9.40625E-07 0.00134  1.07608E-06 0.01931 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39393E-07 0.00123  8.38953E-07 0.00123  9.59423E-07 0.01924 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44031E-03 0.00909  2.40506E-04 0.03236  8.13917E-04 0.01908  6.17751E-04 0.02403  1.38184E-03 0.01344  3.17218E-04 0.03035  6.90738E-05 0.06226 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.03531E-01 0.02382  1.24759E-02 5.8E-05  3.22518E-02 0.00023  1.06726E-01 0.00107  3.00319E-01 0.00048  1.25680E+00 0.00147  6.93716E+00 0.03116 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.28958E-07 0.01963  8.28458E-07 0.01963  9.23162E-07 0.05647 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.39458E-07 0.01962  7.39013E-07 0.01962  8.23258E-07 0.05652 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.20106E-03 0.04411  1.78713E-04 0.14672  6.82117E-04 0.08349  6.27390E-04 0.08510  1.30793E-03 0.06626  3.00346E-04 0.12458  1.04569E-04 0.24552 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.77056E-01 0.11222  1.24757E-02 0.00019  3.22595E-02 0.00073  1.06745E-01 0.00305  3.00253E-01 0.00199  1.25808E+00 0.00354  6.47816E+00 0.09871 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.18670E-03 0.04365  1.83040E-04 0.14673  6.69212E-04 0.08179  6.11816E-04 0.08475  1.31742E-03 0.06525  3.02654E-04 0.12079  1.02550E-04 0.25387 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.73528E-01 0.11092  1.24757E-02 0.00019  3.22573E-02 0.00074  1.06751E-01 0.00303  3.00308E-01 0.00198  1.25759E+00 0.00350  6.44961E+00 0.09953 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.87931E+03 0.04009 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.34069E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33113E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.52477E-03 0.00540 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77430E+03 0.00547 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89363E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71190E-05 0.00776  6.71590E-05 0.00779  1.02336E-06 0.47534 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.56444E-05 0.02495  5.56460E-05 0.02503  8.96843E-07 0.46985 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.86003E-04 0.02279  1.86240E-04 0.02285  1.31294E-04 0.44509 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63641E+01 0.01117 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52201E+01 0.00027  3.98727E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 04:01:30 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01585E+00  1.00309E+00  1.00787E+00  1.01002E+00  9.96022E-01  1.00690E+00  1.00509E+00  1.01208E+00  9.23782E-01  1.00747E+00  1.00112E+00  9.97599E-01  9.98143E-01  1.00143E+00  9.98373E-01  1.00462E+00  1.00314E+00  1.00740E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.10206E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68979E-01 9.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30346E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44702E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70903E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52502E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51520E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.58368E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.38837E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333530 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33400E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33400E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.80876E+03 ;
RUNNING_TIME              (idx, 1)        =  5.24780E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.54433E+00  1.45900E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.21813E+02  5.30262E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.42517E-01  1.82667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.62192E+01  5.40400E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.19417E+02  7.29633E+02 ];
CPU_USAGE                 (idx, 1)        = 9.16338 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98998E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.08891E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.92239E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74199E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.83163E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01426E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39038E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87356E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56854E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.32232E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.18045E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.06780E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.65661E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21212E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20643E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20323E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37667E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34387E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56337E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92503E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11840E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28289E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01389E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.26276E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.13448E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.93964E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96384E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.33472E-02 -7.17548E+27  2.22350E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03400E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.59563E+18 0.00410  1.70401E-02 0.00407 ];
U233_FISS                 (idx, [1:   4]) = [  8.24481E+19 0.00057  8.80465E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.02941E+18 0.00220  6.43875E-02 0.00212 ];
PU239_FISS                (idx, [1:   4]) = [  1.00727E+17 0.01647  1.07616E-03 0.01656 ];
PU240_FISS                (idx, [1:   4]) = [  3.41540E+15 0.08763  3.65340E-05 0.08759 ];
PU241_FISS                (idx, [1:   4]) = [  6.01294E+15 0.06833  6.42237E-05 0.06836 ];
TH232_CAPT                (idx, [1:   4]) = [  9.72888E+19 0.00055  6.69633E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20261E+19 0.00153  8.27757E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.13308E+18 0.00323  1.46831E-02 0.00329 ];
U238_CAPT                 (idx, [1:   4]) = [  4.24306E+14 0.25888  2.91268E-06 0.25886 ];
PU239_CAPT                (idx, [1:   4]) = [  4.73031E+16 0.02375  3.25565E-04 0.02375 ];
PU240_CAPT                (idx, [1:   4]) = [  1.32846E+16 0.04514  9.14197E-05 0.04513 ];
PU241_CAPT                (idx, [1:   4]) = [  1.13530E+15 0.14885  7.81112E-06 0.14888 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84456E+17 0.01180  1.26956E-03 0.01178 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002007 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09349E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002007 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5484565 5.48968E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3535217 3.53827E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 982225 9.82979E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002007 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.12342E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36158E+20 4.4E-06  2.36158E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36720E+19 1.0E-07  9.36720E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45286E+20 0.00028  1.28962E+20 0.00019  1.63239E+19 0.00169 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38958E+20 0.00017  2.22634E+20 0.00011  1.63239E+19 0.00169 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64655E+20 0.00036  2.64655E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67504E+22 0.00024  3.47841E+22 0.00022  1.96625E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60156E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64974E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19622E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  8.11812E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.11812E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.67296E-01 0.05000 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21858E-01 0.01273 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90375E-04 0.02349 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.53495E+03 0.02811 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01705E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.08697E-01 0.03637 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.39062E-01 0.03637 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52111E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99895E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.92233E-01 0.00045  9.87769E-02 0.00045  3.40288E-04 0.00881 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92246E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92359E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92246E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89503E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90442E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90292E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50550E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50748E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.94735E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.94568E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94011E-03 0.00555  2.82140E-04 0.01909  9.24137E-04 0.01072  7.22324E-04 0.01208  1.57317E-03 0.00884  3.56330E-04 0.01782  8.20023E-05 0.03724 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10827E-01 0.01475  1.24351E-02 0.00334  3.22477E-02 0.00013  1.06703E-01 0.00058  3.00755E-01 0.00032  1.25715E+00 0.00081  6.44490E+00 0.02621 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44390E-03 0.00795  2.35935E-04 0.02902  7.96716E-04 0.01590  6.25649E-04 0.01802  1.39412E-03 0.01267  3.16067E-04 0.02663  7.54114E-05 0.05496 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.19414E-01 0.02183  1.24776E-02 5.7E-05  3.22508E-02 0.00018  1.06617E-01 0.00079  3.00641E-01 0.00044  1.25731E+00 0.00106  7.09634E+00 0.02118 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.33379E-07 0.00126  9.32912E-07 0.00127  1.06942E-06 0.01734 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32750E-07 0.00121  8.32333E-07 0.00121  9.54231E-07 0.01738 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42423E-03 0.00892  2.40007E-04 0.03327  8.10231E-04 0.01899  6.27157E-04 0.02131  1.37411E-03 0.01465  3.08434E-04 0.03050  6.42965E-05 0.06679 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.01361E-01 0.02611  1.24766E-02 4.7E-05  3.22400E-02 0.00025  1.06532E-01 0.00097  3.00825E-01 0.00055  1.25471E+00 0.00138  7.33550E+00 0.03017 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.24682E-07 0.01975  8.24141E-07 0.01975  9.89106E-07 0.06142 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.35884E-07 0.01974  7.35400E-07 0.01975  8.82775E-07 0.06137 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87824E-03 0.04333  2.10612E-04 0.13577  6.53656E-04 0.08187  4.98223E-04 0.09447  1.20071E-03 0.06800  2.73697E-04 0.13402  4.13498E-05 0.29706 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.62443E-01 0.08716  1.24765E-02 0.00015  3.22327E-02 0.00078  1.06922E-01 0.00356  3.00282E-01 0.00183  1.25744E+00 0.00419  8.32988E+00 0.08057 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87840E-03 0.04292  2.14352E-04 0.13220  6.58348E-04 0.08163  4.99917E-04 0.09312  1.19477E-03 0.06647  2.70647E-04 0.13064  4.03713E-05 0.30547 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.63114E-01 0.08899  1.24765E-02 0.00015  3.22325E-02 0.00077  1.06947E-01 0.00354  3.00146E-01 0.00181  1.25699E+00 0.00438  8.32988E+00 0.08057 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.52064E+03 0.03890 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.25090E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.25345E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39244E-03 0.00669 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.66714E+03 0.00666 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.88363E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74306E-05 0.00655  6.74323E-05 0.00655  1.49531E-06 0.41110 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.60399E-05 0.02694  5.59991E-05 0.02739  2.06959E-06 0.52682 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.82053E-04 0.02250  1.82169E-04 0.02252  1.51090E-04 0.40532 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58936E+01 0.01083 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51520E+01 0.00027  3.96988E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 04:55:12 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01826E+00  1.00533E+00  1.00304E+00  1.00656E+00  1.00087E+00  1.00628E+00  9.94357E-01  1.00908E+00  9.25811E-01  9.97371E-01  9.99042E-01  1.00409E+00  1.00862E+00  1.00239E+00  9.96754E-01  1.00412E+00  1.00714E+00  1.01088E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.12309E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68769E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29984E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44422E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71046E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51198E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50215E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.56950E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.39943E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333607 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33379E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33379E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.29628E+03 ;
RUNNING_TIME              (idx, 1)        =  5.78480E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.69525E+00  1.50917E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.75310E+02  5.34972E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.60900E-01  1.83833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.15444E+01  5.32517E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.73196E+02  7.32782E+02 ];
CPU_USAGE                 (idx, 1)        = 9.15552 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99014E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.08004E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.93937E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74742E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.10403E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97564E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36725E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92917E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57631E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.80957E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75755E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.31934E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04483E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44244E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70329E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34119E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37174E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33747E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55613E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.96884E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27132E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27581E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.06452E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.42109E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10295E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.95892E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60422E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.43749E-02 -1.17001E+28  2.26875E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03276E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.55048E+18 0.00409  1.65861E-02 0.00407 ];
U233_FISS                 (idx, [1:   4]) = [  8.03789E+19 0.00056  8.59834E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.93417E+18 0.00187  7.41799E-02 0.00188 ];
U238_FISS                 (idx, [1:   4]) = [  5.37915E+13 0.70601  5.74684E-07 0.70604 ];
PU239_FISS                (idx, [1:   4]) = [  3.27243E+17 0.00857  3.50112E-03 0.00860 ];
PU240_FISS                (idx, [1:   4]) = [  2.40519E+16 0.03495  2.57211E-04 0.03490 ];
PU241_FISS                (idx, [1:   4]) = [  3.85632E+16 0.02626  4.12444E-04 0.02621 ];
TH232_CAPT                (idx, [1:   4]) = [  9.52018E+19 0.00054  6.52046E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17367E+19 0.00145  8.03874E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.44529E+18 0.00330  1.67487E-02 0.00331 ];
U238_CAPT                 (idx, [1:   4]) = [  1.16629E+15 0.15702  7.99018E-06 0.15695 ];
PU239_CAPT                (idx, [1:   4]) = [  1.51907E+17 0.01291  1.04062E-03 0.01294 ];
PU240_CAPT                (idx, [1:   4]) = [  6.77077E+16 0.01943  4.63805E-04 0.01945 ];
PU241_CAPT                (idx, [1:   4]) = [  7.49175E+15 0.05956  5.13061E-05 0.05959 ];
SM149_CAPT                (idx, [1:   4]) = [  1.87370E+17 0.01274  1.28316E-03 0.01270 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001375 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08178E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001375 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498112 5.50346E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3520412 3.52368E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 982851 9.83681E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001375 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.02914E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36329E+20 4.5E-06  2.36329E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36324E+19 1.0E-07  9.36324E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45905E+20 0.00031  1.29407E+20 0.00020  1.64977E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39537E+20 0.00019  2.23039E+20 0.00011  1.64977E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65297E+20 0.00034  2.65297E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68425E+22 0.00025  3.48612E+22 0.00022  1.98130E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60976E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65635E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19564E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.99353E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.99353E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.64707E-01 0.04589 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.39874E-01 0.01081 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.24245E-04 0.02130 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.01112E+03 0.02912 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01635E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.72380E-01 0.03042 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.96454E-01 0.03042 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52401E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99979E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.89286E-01 0.00044  9.84752E-02 0.00044  3.43228E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90663E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.90837E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90663E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87816E-01 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.86260E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.86160E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.56975E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.57105E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.03416E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.04034E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.00745E-03 0.00564  2.79468E-04 0.02225  9.49256E-04 0.01150  7.23128E-04 0.01269  1.60358E-03 0.00828  3.64051E-04 0.01656  8.79697E-05 0.03452 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.15476E-01 0.01415  1.24786E-02 9.1E-05  3.22183E-02 0.00014  1.06666E-01 0.00055  3.01214E-01 0.00031  1.25971E+00 0.00087  6.50480E+00 0.02382 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48094E-03 0.00796  2.50390E-04 0.03245  8.17528E-04 0.01612  6.28062E-04 0.01727  1.39465E-03 0.01117  3.04725E-04 0.02388  8.55839E-05 0.05005 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.31745E-01 0.02127  1.24775E-02 6.2E-05  3.22231E-02 0.00021  1.06577E-01 0.00069  3.01142E-01 0.00043  1.25831E+00 0.00127  7.07508E+00 0.02033 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.21370E-07 0.00145  9.20856E-07 0.00145  1.06495E-06 0.02222 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.19304E-07 0.00135  8.18846E-07 0.00135  9.47208E-07 0.02233 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45997E-03 0.00917  2.32474E-04 0.03669  8.09305E-04 0.01928  6.28910E-04 0.02041  1.39231E-03 0.01442  3.15887E-04 0.03067  8.10912E-05 0.05976 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.25727E-01 0.02627  1.24790E-02 0.00018  3.22213E-02 0.00028  1.06785E-01 0.00101  3.01368E-01 0.00053  1.25707E+00 0.00159  6.72970E+00 0.03006 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.13239E-07 0.01976  8.12830E-07 0.01977  8.72172E-07 0.05347 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23283E-07 0.01975  7.22920E-07 0.01976  7.75673E-07 0.05342 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.26237E-03 0.04340  2.15537E-04 0.13728  7.83127E-04 0.07979  5.76168E-04 0.09368  1.28419E-03 0.06561  3.10670E-04 0.12525  9.26781E-05 0.23666 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.66680E-01 0.10947  1.24851E-02 0.00064  3.22132E-02 0.00097  1.06386E-01 0.00300  3.02453E-01 0.00213  1.25959E+00 0.00607  6.93402E+00 0.07943 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.22788E-03 0.04274  2.14014E-04 0.13410  7.73820E-04 0.07845  5.80003E-04 0.09362  1.26265E-03 0.06426  3.08179E-04 0.12203  8.92131E-05 0.23624 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.64634E-01 0.10756  1.24822E-02 0.00041  3.22132E-02 0.00097  1.06414E-01 0.00298  3.02425E-01 0.00212  1.26019E+00 0.00586  6.93515E+00 0.07940 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.04987E+03 0.03918 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.13574E-07 0.00083 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12378E-07 0.00068 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.49385E-03 0.00538 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82478E+03 0.00541 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92360E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69554E-05 0.00646  6.69382E-05 0.00646  9.59796E-07 0.51218 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.69057E-05 0.02512  5.69624E-05 0.02509  2.07048E-07 0.67366 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.14302E-04 0.02063  2.14759E-04 0.02064  1.02542E-04 0.50428 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61908E+01 0.01153 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50215E+01 0.00027  3.93107E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 05:49:04 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01426E+00  1.00723E+00  1.01032E+00  1.00430E+00  9.97729E-01  1.00525E+00  1.01027E+00  1.00469E+00  9.29278E-01  9.99532E-01  1.00523E+00  9.99924E-01  1.00339E+00  1.00686E+00  9.99302E-01  1.00059E+00  9.96935E-01  1.00490E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.17819E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68218E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.28650E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.43311E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72205E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51177E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50186E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61125E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.44057E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333911 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33384E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33384E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.78577E+03 ;
RUNNING_TIME              (idx, 1)        =  6.32346E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.84675E+00  1.51500E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.28969E+02  5.36583E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.79717E-01  1.88167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.68319E+01  5.28743E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.27099E+02  7.34434E+02 ];
CPU_USAGE                 (idx, 1)        = 9.14969 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98997E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.07345E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.92254E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74172E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.00169E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92930E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34251E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95870E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57310E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.49352E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38436E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.31884E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13499E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69250E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23865E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57858E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34631E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31714E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52886E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00181E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59801E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24020E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04146E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.11188E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.03101E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.98645E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32052E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.09119E-01 -2.34797E+28  2.38654E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02562E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.52956E+18 0.00415  1.63360E-02 0.00406 ];
U233_FISS                 (idx, [1:   4]) = [  7.93038E+19 0.00056  8.47033E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  7.13035E+18 0.00203  7.61582E-02 0.00196 ];
U238_FISS                 (idx, [1:   4]) = [  2.12236E+14 0.34944  2.27825E-06 0.34943 ];
PU239_FISS                (idx, [1:   4]) = [  6.15940E+17 0.00641  6.57835E-03 0.00636 ];
PU240_FISS                (idx, [1:   4]) = [  7.50213E+16 0.01940  8.01048E-04 0.01933 ];
PU241_FISS                (idx, [1:   4]) = [  1.33359E+17 0.01520  1.42428E-03 0.01515 ];
TH232_CAPT                (idx, [1:   4]) = [  9.33613E+19 0.00056  6.37435E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15104E+19 0.00150  7.85896E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51277E+18 0.00325  1.71565E-02 0.00324 ];
U238_CAPT                 (idx, [1:   4]) = [  2.49882E+15 0.10779  1.70601E-05 0.10777 ];
PU239_CAPT                (idx, [1:   4]) = [  2.90612E+17 0.00897  1.98421E-03 0.00896 ];
PU240_CAPT                (idx, [1:   4]) = [  1.95080E+17 0.01102  1.33192E-03 0.01102 ];
PU241_CAPT                (idx, [1:   4]) = [  2.82275E+16 0.03120  1.92735E-04 0.03121 ];
SM149_CAPT                (idx, [1:   4]) = [  1.91248E+17 0.01214  1.30575E-03 0.01213 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001516 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07038E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001516 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496440 5.50174E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3513878 3.51694E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 991198 9.92025E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001516 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.13621E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36623E+20 4.9E-06  2.36623E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35944E+19 9.5E-08  9.35944E+19 9.5E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46625E+20 0.00029  1.29519E+20 0.00021  1.71064E+19 0.00174 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40220E+20 0.00018  2.23113E+20 0.00012  1.71064E+19 0.00174 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66215E+20 0.00035  2.66215E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.72134E+22 0.00025  3.51919E+22 0.00021  2.02140E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.64099E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66630E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19969E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.74730E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.74730E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.49488E-01 0.03890 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.11227E-01 0.01090 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.91638E-04 0.01900 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.09012E+03 0.03138 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.00802E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.29309E-01 0.02524 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.47095E-01 0.02524 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52818E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00060E+02 9.5E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.89187E-01 0.00045  9.84531E-02 0.00044  3.45308E-04 0.00919 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.88426E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88875E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.88426E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86212E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83452E+00 0.00023 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83522E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61459E-02 0.00160 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61305E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10984E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10468E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03849E-03 0.00546  2.86834E-04 0.01990  9.37469E-04 0.01101  7.24516E-04 0.01249  1.62660E-03 0.00866  3.78009E-04 0.01669  8.50689E-05 0.03504 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.11073E-01 0.01383  1.24820E-02 0.00010  3.22077E-02 0.00015  1.06704E-01 0.00056  3.01352E-01 0.00030  1.25930E+00 0.00094  6.31357E+00 0.02494 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51281E-03 0.00777  2.44664E-04 0.02915  8.06506E-04 0.01634  6.45076E-04 0.01698  1.41810E-03 0.01272  3.24977E-04 0.02582  7.34866E-05 0.05333 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.07710E-01 0.02057  1.24839E-02 0.00017  3.22120E-02 0.00023  1.06782E-01 0.00077  3.01275E-01 0.00042  1.25822E+00 0.00142  6.74357E+00 0.02168 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.16432E-07 0.00132  9.15824E-07 0.00129  1.08840E-06 0.03539 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14838E-07 0.00126  8.14299E-07 0.00124  9.67112E-07 0.03501 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.49655E-03 0.00921  2.48410E-04 0.03333  8.10793E-04 0.01825  6.11299E-04 0.02188  1.43336E-03 0.01488  3.21777E-04 0.03049  7.09112E-05 0.06463 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04297E-01 0.02472  1.24867E-02 0.00033  3.22123E-02 0.00026  1.06801E-01 0.00110  3.01188E-01 0.00050  1.25962E+00 0.00182  6.65528E+00 0.03305 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.05218E-07 0.01983  8.04791E-07 0.01983  8.52380E-07 0.05343 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.16112E-07 0.01981  7.15735E-07 0.01982  7.57668E-07 0.05335 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13071E-03 0.04742  2.08355E-04 0.14454  7.27903E-04 0.09505  5.32429E-04 0.10435  1.31616E-03 0.06452  3.04083E-04 0.13630  4.17847E-05 0.25096 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.01068E-01 0.10420  1.24899E-02 0.00096  3.21843E-02 0.00112  1.07241E-01 0.00372  3.01020E-01 0.00190  1.26046E+00 0.00562  7.43213E+00 0.08082 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.11620E-03 0.04689  2.05786E-04 0.14194  7.29732E-04 0.09443  5.27258E-04 0.10212  1.31144E-03 0.06414  3.00507E-04 0.13458  4.14774E-05 0.25012 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.08163E-01 0.10691  1.24899E-02 0.00096  3.21867E-02 0.00111  1.07243E-01 0.00370  3.00963E-01 0.00189  1.26011E+00 0.00561  7.43213E+00 0.08082 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.91490E+03 0.04357 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.06881E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.06334E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47466E-03 0.00613 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.83064E+03 0.00596 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.08161E-09 0.00088 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76838E-05 0.00539  6.76760E-05 0.00541  2.34442E-06 0.31880 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.86218E-05 0.02177  5.85872E-05 0.02180  1.98546E-06 0.42299 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.77681E-04 0.01884  2.77708E-04 0.01891  2.87083E-04 0.32281 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59190E+01 0.01035 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50186E+01 0.00027  3.90310E+01 0.00041 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 06:43:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02457E+00  1.01169E+00  1.00562E+00  1.00354E+00  9.99752E-01  1.01187E+00  9.99512E-01  1.00032E+00  9.30512E-01  9.94040E-01  1.00567E+00  9.96387E-01  9.97807E-01  1.00364E+00  1.00203E+00  1.00434E+00  1.00419E+00  1.00451E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.24465E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67553E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.27098E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.42034E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73773E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51273E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50269E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.66182E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.48939E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334153 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33403E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33403E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.27696E+03 ;
RUNNING_TIME              (idx, 1)        =  6.86395E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.00055E+00  1.53800E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.82806E+02  5.38373E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.98050E-01  1.83333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.21284E+01  5.29650E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.81140E+02  7.34971E+02 ];
CPU_USAGE                 (idx, 1)        = 9.14483 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98996E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.06797E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.91926E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74015E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.38876E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91673E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33566E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96789E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57211E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57478E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55546E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39254E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14767E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76508E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39650E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93116E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33997E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31253E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52236E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01712E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00913E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22974E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03155E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27233E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01078E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.99876E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98062E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.65007E-01 -3.55053E+28  2.50680E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01985E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.53603E+18 0.00436  1.64084E-02 0.00435 ];
U233_FISS                 (idx, [1:   4]) = [  7.92157E+19 0.00054  8.46197E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.07604E+18 0.00202  7.55870E-02 0.00194 ];
U238_FISS                 (idx, [1:   4]) = [  7.92359E+13 0.57543  8.41647E-07 0.57542 ];
PU239_FISS                (idx, [1:   4]) = [  6.59472E+17 0.00598  7.04428E-03 0.00593 ];
PU240_FISS                (idx, [1:   4]) = [  8.37424E+16 0.01698  8.94296E-04 0.01690 ];
PU241_FISS                (idx, [1:   4]) = [  1.52083E+17 0.01277  1.62473E-03 0.01277 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26146E+19 0.00054  6.32147E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15420E+19 0.00145  7.87819E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49173E+18 0.00347  1.70068E-02 0.00341 ];
U238_CAPT                 (idx, [1:   4]) = [  2.89782E+15 0.09523  1.97952E-05 0.09519 ];
PU239_CAPT                (idx, [1:   4]) = [  3.08227E+17 0.00935  2.10380E-03 0.00934 ];
PU240_CAPT                (idx, [1:   4]) = [  2.23617E+17 0.01126  1.52627E-03 0.01124 ];
PU241_CAPT                (idx, [1:   4]) = [  3.33663E+16 0.02789  2.27748E-04 0.02787 ];
SM149_CAPT                (idx, [1:   4]) = [  1.97288E+17 0.01242  1.34651E-03 0.01239 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002078 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08773E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002078 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5489853 5.49495E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3508139 3.51108E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1004086 1.00485E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002078 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.28991E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36680E+20 4.5E-06  2.36680E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35893E+19 1.0E-07  9.35893E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46481E+20 0.00026  1.28963E+20 0.00018  1.75181E+19 0.00166 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40070E+20 0.00016  2.22552E+20 0.00010  1.75181E+19 0.00166 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66625E+20 0.00034  2.66625E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.75513E+22 0.00025  3.55128E+22 0.00022  2.03852E+21 0.00168 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.67925E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66862E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20178E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.53758E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.53758E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.14310E-01 0.03751 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.05375E-01 0.00857 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.71221E-04 0.01803 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.38822E+03 0.03399 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99521E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.65454E-01 0.02170 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.78432E-01 0.02170 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52892E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.87930E-01 0.00041  9.83199E-02 0.00039  3.39187E-04 0.00847 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.87869E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.87718E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.87869E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87073E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83282E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83261E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61720E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61728E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11346E-01 0.00135 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11520E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.00582E-03 0.00555  2.90004E-04 0.01923  9.29054E-04 0.01135  7.30081E-04 0.01265  1.59913E-03 0.00840  3.73252E-04 0.01771  8.43059E-05 0.03585 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13669E-01 0.01439  1.24771E-02 5.8E-05  3.21972E-02 0.00015  1.06832E-01 0.00057  3.01314E-01 0.00032  1.25690E+00 0.00106  6.39046E+00 0.02580 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49630E-03 0.00758  2.44057E-04 0.02798  8.07997E-04 0.01659  6.42762E-04 0.01816  1.39851E-03 0.01216  3.30331E-04 0.02756  7.26456E-05 0.05349 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09555E-01 0.02038  1.24768E-02 7.4E-05  3.22019E-02 0.00022  1.06774E-01 0.00075  3.01503E-01 0.00041  1.26060E+00 0.00120  6.95483E+00 0.02054 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.24629E-07 0.00142  9.24170E-07 0.00142  1.05555E-06 0.01924 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.20967E-07 0.00136  8.20559E-07 0.00136  9.37317E-07 0.01926 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43429E-03 0.00851  2.41643E-04 0.03533  8.19142E-04 0.01848  6.12352E-04 0.02219  1.38425E-03 0.01416  3.08917E-04 0.03003  6.79877E-05 0.06376 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.98901E-01 0.02461  1.24774E-02 9.2E-05  3.21984E-02 0.00025  1.06778E-01 0.00102  3.01092E-01 0.00051  1.25611E+00 0.00228  6.85781E+00 0.03231 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.08431E-07 0.01988  8.07931E-07 0.01988  9.02366E-07 0.05414 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.17972E-07 0.01988  7.17527E-07 0.01989  8.01367E-07 0.05415 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16333E-03 0.04321  2.58694E-04 0.11599  6.60061E-04 0.08598  5.59624E-04 0.09083  1.38494E-03 0.06661  2.48418E-04 0.13550  5.15976E-05 0.27496 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.83212E-01 0.10744  1.24780E-02 9.7E-05  3.22198E-02 0.00100  1.06668E-01 0.00308  3.00886E-01 0.00197  1.24910E+00 0.00679  6.09193E+00 0.11834 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13828E-03 0.04276  2.61991E-04 0.11404  6.59406E-04 0.08607  5.52989E-04 0.09000  1.36189E-03 0.06548  2.47961E-04 0.13289  5.40372E-05 0.27520 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.87933E-01 0.10540  1.24779E-02 9.8E-05  3.22201E-02 0.00100  1.06661E-01 0.00307  3.00916E-01 0.00196  1.24905E+00 0.00679  6.09193E+00 0.11834 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.94755E+03 0.03967 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.14704E-07 0.00081 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12148E-07 0.00067 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42916E-03 0.00568 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.74957E+03 0.00573 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.27142E-09 0.00092 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.86717E-05 0.00493  6.86582E-05 0.00495  1.96343E-06 0.36342 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.99333E-05 0.01599  5.98723E-05 0.01591  2.29995E-06 0.53292 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.54334E-04 0.01770  3.54957E-04 0.01773  2.04406E-04 0.35081 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62575E+01 0.01112 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50269E+01 0.00026  3.89850E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 21])  = 'galo.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 19 19:16:43 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 07:12:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650406603631 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02593E+00  1.00153E+00  1.01071E+00  1.00289E+00  9.97311E-01  1.00942E+00  1.00011E+00  1.00772E+00  8.80036E-01  1.00174E+00  1.01237E+00  1.00447E+00  1.00580E+00  1.00510E+00  1.00070E+00  1.01286E+00  1.00286E+00  1.01844E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.31299E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66870E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.25834E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.41052E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.74995E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52315E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51302E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.71965E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.54031E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333731 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33407E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33407E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.54498E+03 ;
RUNNING_TIME              (idx, 1)        =  7.15750E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.99617E-01  9.99617E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.15975E+00  1.59200E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.11944E+02  2.91376E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.17650E-01  1.96000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.31000E-02  1.31000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.47894E+01  2.66087E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.13132E+02  7.13132E+02 ];
CPU_USAGE                 (idx, 1)        = 9.14422 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98352E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.06782E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22649.81;
MEMSIZE                   (idx, 1)        = 22452.40;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
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

TOT_ACTIVITY              (idx, 1)        =  6.91348E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73959E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.56070E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90706E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32959E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.97172E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57209E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.58411E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60182E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39959E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14628E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78654E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44400E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03310E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33889E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31185E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52132E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02446E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13647E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22263E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02950E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.30007E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99253E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  8.00689E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64071E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.21333E-01 -4.76251E+28  2.62800E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01370E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.53560E+18 0.00388  1.64021E-02 0.00384 ];
U233_FISS                 (idx, [1:   4]) = [  7.92173E+19 0.00058  8.46142E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.08103E+18 0.00191  7.56367E-02 0.00189 ];
U238_FISS                 (idx, [1:   4]) = [  1.06261E+14 0.49752  1.13135E-06 0.49752 ];
PU239_FISS                (idx, [1:   4]) = [  6.61786E+17 0.00663  7.06928E-03 0.00664 ];
PU240_FISS                (idx, [1:   4]) = [  8.66096E+16 0.01624  9.25014E-04 0.01620 ];
PU241_FISS                (idx, [1:   4]) = [  1.57488E+17 0.01361  1.68200E-03 0.01358 ];
TH232_CAPT                (idx, [1:   4]) = [  9.20368E+19 0.00053  6.28182E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15166E+19 0.00148  7.86057E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49476E+18 0.00324  1.70280E-02 0.00323 ];
U238_CAPT                 (idx, [1:   4]) = [  3.36288E+15 0.09702  2.29756E-05 0.09709 ];
PU239_CAPT                (idx, [1:   4]) = [  3.08080E+17 0.00901  2.10296E-03 0.00904 ];
PU240_CAPT                (idx, [1:   4]) = [  2.24818E+17 0.01126  1.53458E-03 0.01129 ];
PU241_CAPT                (idx, [1:   4]) = [  3.33748E+16 0.02857  2.27789E-04 0.02857 ];
SM149_CAPT                (idx, [1:   4]) = [  2.02168E+17 0.01098  1.37997E-03 0.01100 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002205 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08596E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002205 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5484527 5.48953E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3504903 3.50779E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1012775 1.01353E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002205 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.60187E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36687E+20 4.7E-06  2.36687E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35891E+19 1.0E-07  9.35891E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46558E+20 0.00030  1.28474E+20 0.00019  1.80839E+19 0.00166 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40147E+20 0.00018  2.22063E+20 0.00011  1.80839E+19 0.00166 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66896E+20 0.00032  2.66896E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.79043E+22 0.00024  3.58290E+22 0.00022  2.07525E+21 0.00161 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.70513E+19 0.00134 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67198E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20575E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29424E+04 ;
TOT_FMASS                 (idx, 1)        =  7.36223E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29424E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.36223E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.13017E-01 0.03075 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.96105E-01 0.00790 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.72126E-04 0.01598 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.38711E+03 0.03437 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.98656E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.27930E-01 0.01462 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.33883E-01 0.01462 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52900E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.87008E-01 0.00044  9.82234E-02 0.00043  3.45394E-04 0.00910 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.86777E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.86839E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.86777E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86784E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83727E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83622E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.60999E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61142E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11371E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11482E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03888E-03 0.00543  2.86678E-04 0.01947  9.26448E-04 0.01060  7.20239E-04 0.01267  1.64642E-03 0.00854  3.72444E-04 0.01762  8.66535E-05 0.03627 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.14216E-01 0.01412  1.24803E-02 0.00013  3.22091E-02 0.00016  1.06770E-01 0.00055  3.01420E-01 0.00030  1.25583E+00 0.00109  6.53069E+00 0.02485 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49440E-03 0.00731  2.50284E-04 0.02758  7.97729E-04 0.01565  6.35607E-04 0.01801  1.40719E-03 0.01193  3.29860E-04 0.02392  7.37288E-05 0.05173 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.11323E-01 0.01915  1.24797E-02 0.00011  3.22059E-02 0.00023  1.06674E-01 0.00076  3.01715E-01 0.00045  1.25668E+00 0.00118  6.94725E+00 0.02149 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.40076E-07 0.00154  9.39706E-07 0.00153  1.04625E-06 0.01852 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.33802E-07 0.00147  8.33473E-07 0.00146  9.27965E-07 0.01853 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.51924E-03 0.00925  2.52973E-04 0.03372  8.12569E-04 0.01819  6.40669E-04 0.02148  1.43102E-03 0.01367  3.09192E-04 0.03094  7.28155E-05 0.06349 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.01920E-01 0.02488  1.24793E-02 0.00016  3.21967E-02 0.00029  1.06886E-01 0.00104  3.01354E-01 0.00055  1.25563E+00 0.00189  6.92708E+00 0.03051 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.21138E-07 0.01979  8.20830E-07 0.01979  8.83503E-07 0.05942 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.28537E-07 0.01979  7.28264E-07 0.01979  7.83840E-07 0.05934 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06806E-03 0.04469  2.28223E-04 0.14334  7.52042E-04 0.07898  5.00984E-04 0.10070  1.22865E-03 0.06761  2.78851E-04 0.12307  7.93054E-05 0.23560 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.53587E-01 0.09751  1.24903E-02 0.00097  3.21276E-02 0.00117  1.06634E-01 0.00315  3.02495E-01 0.00209  1.26847E+00 0.00426  7.36322E+00 0.07779 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.04527E-03 0.04433  2.25949E-04 0.14378  7.39741E-04 0.07795  5.04063E-04 0.09918  1.21369E-03 0.06675  2.81839E-04 0.12111  7.99905E-05 0.23542 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.49711E-01 0.09243  1.24902E-02 0.00097  3.21286E-02 0.00118  1.06619E-01 0.00312  3.02454E-01 0.00207  1.26862E+00 0.00427  7.36322E+00 0.07779 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.73462E+03 0.03974 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.26401E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.21679E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.54996E-03 0.00597 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.83318E+03 0.00610 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.50031E-09 0.00107 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.97273E-05 0.00426  6.96985E-05 0.00426  4.04053E-06 0.25798 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.26147E-05 0.01555  6.26107E-05 0.01561  2.95227E-06 0.30862 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.54057E-04 0.01572  4.54192E-04 0.01584  4.16621E-04 0.24756 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58290E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51302E+01 0.00026  3.90365E+01 0.00035 ];

