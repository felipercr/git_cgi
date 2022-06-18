
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 12:58:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01554E+00  1.00040E+00  9.90725E-01  1.00313E+00  9.99137E-01  9.98811E-01  9.95558E-01  9.97380E-01  1.00184E+00  1.00520E+00  9.95297E-01  1.00301E+00  1.00180E+00  9.97877E-01  9.97746E-01  1.00143E+00  9.88125E-01  1.00700E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.14278E-02 0.00037  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58572E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39440E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53326E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66349E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46805E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45899E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16109E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.77366E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500312 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50075E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50075E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.47928E+02 ;
RUNNING_TIME              (idx, 1)        =  2.59469E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.88400E-01  1.88400E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.45798E+01  2.45798E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.48800E-01  8.01333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.59061E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.55521 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98000E+00 3.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.53726E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  2.79821E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.20056E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13472E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.79821E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.20056E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.68943E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.42765E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.68943E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.42765E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29734E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.13375E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.31792E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.46505E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.69002E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.53541E+18 0.00385  1.63771E-02 0.00383 ];
U233_FISS                 (idx, [1:   4]) = [  9.22199E+19 0.00050  9.83623E-01 6.4E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.18389E+19 0.00053  7.54828E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34629E+19 0.00133  1.10652E-01 0.00124 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002994 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13723E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002994 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5137373 5.14184E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3958984 3.96222E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906637 9.07321E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002994 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.32248E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36437E+20 3.6E-06  2.36437E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.3E-07  9.37766E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21606E+20 0.00033  1.08335E+20 0.00025  1.32713E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15382E+20 0.00018  2.02111E+20 0.00013  1.32713E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36626E+20 0.00035  2.36626E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.13164E+22 0.00026  2.97001E+22 0.00024  1.61633E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.14703E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.36853E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05625E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.32045E-01 0.07813 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.79485E-01 0.00870 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.01032E-04 0.01477 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.83563E+03 0.02137 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09278E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.98343E-01 0.06639 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.62340E-01 0.06639 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52128E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.98874E-01 0.00043  6.22331E-02 0.00042  2.02953E-04 0.00854 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.99415E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.99251E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.99415E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09916E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04879E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04886E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30313E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30285E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.46240E-01 0.00163 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46462E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.23197E-03 0.00529  2.58067E-04 0.02024  7.87540E-04 0.01153  5.88946E-04 0.01294  1.26958E-03 0.00877  2.78022E-04 0.01938  4.98149E-05 0.04714 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47281E-01 0.01547  1.24758E-02 2.8E-05  3.23296E-02 8.7E-05  1.06054E-01 0.00060  2.97429E-01 0.00027  1.23534E+00 0.00024  4.38081E+00 0.04301 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26140E-03 0.00821  2.63297E-04 0.02972  7.84055E-04 0.01692  5.90372E-04 0.01921  1.28736E-03 0.01316  2.83547E-04 0.02857  5.27677E-05 0.06592 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56676E-01 0.02305  1.24759E-02 3.7E-05  3.23342E-02 0.00013  1.05892E-01 0.00075  2.97274E-01 0.00037  1.23517E+00 0.00030  6.31638E+00 0.02936 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.35693E-07 0.00123  8.35333E-07 0.00123  9.43379E-07 0.01667 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.34693E-07 0.00116  8.34333E-07 0.00116  9.42295E-07 0.01669 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24660E-03 0.00874  2.65137E-04 0.03040  8.10098E-04 0.01883  5.76987E-04 0.02198  1.26168E-03 0.01401  2.81622E-04 0.03159  5.10809E-05 0.07358 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.46898E-01 0.02447  1.24755E-02 4.9E-05  3.23298E-02 0.00015  1.05819E-01 0.00090  2.97434E-01 0.00047  1.23540E+00 0.00037  6.12545E+00 0.04102 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.39883E-07 0.01701  7.39483E-07 0.01700  8.43085E-07 0.06755 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.39351E-07 0.01701  7.38952E-07 0.01701  8.42640E-07 0.06749 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93271E-03 0.04149  2.20174E-04 0.12869  7.37489E-04 0.07455  4.86522E-04 0.08795  1.18760E-03 0.06090  2.43228E-04 0.12605  5.77049E-05 0.27971 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52476E-01 0.07791  1.24736E-02 0.00017  3.23330E-02 0.00048  1.05299E-01 0.00208  2.97335E-01 0.00150  1.23647E+00 0.00100  5.49175E+00 0.13124 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93554E-03 0.04071  2.19114E-04 0.12897  7.43461E-04 0.07350  4.75490E-04 0.08653  1.20279E-03 0.05967  2.36206E-04 0.12146  5.84815E-05 0.27958 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.53810E-01 0.08043  1.24736E-02 0.00017  3.23338E-02 0.00048  1.05296E-01 0.00206  2.97393E-01 0.00150  1.23648E+00 0.00100  5.51534E+00 0.13082 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.98967E+03 0.03836 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.29976E-07 0.00062 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.28982E-07 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24493E-03 0.00584 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90997E+03 0.00582 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.25554E-09 0.00132 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.25978E-05 0.00399  7.25967E-05 0.00399  1.82392E-06 0.32431 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.87117E-05 0.01546  7.86670E-05 0.01550  2.71166E-06 0.42507 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.84605E-04 0.01418  4.85079E-04 0.01417  3.43709E-04 0.32822 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68770E+01 0.01164 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45899E+01 0.00028  4.04393E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 13:34:59 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01030E+00  9.99920E-01  1.00775E+00  9.94191E-01  9.95445E-01  9.94158E-01  1.00008E+00  1.00237E+00  1.00547E+00  1.00347E+00  9.97196E-01  9.97673E-01  9.89148E-01  1.00103E+00  1.00675E+00  9.90813E-01  9.98921E-01  1.00530E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.13601E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58640E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39340E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53206E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66545E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47439E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46529E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17693E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.77417E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500592 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50063E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50063E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.14608E+02 ;
RUNNING_TIME              (idx, 1)        =  6.27888E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.12867E-01  1.24467E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.12708E+01  3.66910E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.80167E-02  1.80167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.98767E-01  4.99167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.27817E+01  3.84171E+02 ];
CPU_USAGE                 (idx, 1)        = 9.78849 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98507E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.78255E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  5.76518E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60320E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65520E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.46983E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.03540E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26472E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46895E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.44943E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65348E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.38629E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.75355E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06314E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17812E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44279E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35672E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17525E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55420E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59263E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38167E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.38995E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51296E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50591E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.28285E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.55595E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08112E-04 -2.38708E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.79655E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.55007E+18 0.00402  1.65359E-02 0.00399 ];
U233_FISS                 (idx, [1:   4]) = [  9.21616E+19 0.00051  9.83165E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.03251E+14 0.23764  4.29872E-06 0.23765 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28230E+19 0.00050  7.50592E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34282E+19 0.00132  1.08583E-01 0.00122 ];
U235_CAPT                 (idx, [1:   4]) = [  9.68291E+13 0.49817  7.82162E-07 0.49813 ];
SM149_CAPT                (idx, [1:   4]) = [  1.24314E+16 0.04388  1.00596E-04 0.04396 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002520 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14646E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002520 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5171775 5.17658E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3920406 3.92385E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910339 9.11030E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002520 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.99303E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36431E+20 3.6E-06  2.36431E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37763E+19 1.2E-07  9.37763E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23722E+20 0.00031  1.10317E+20 0.00022  1.34045E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17498E+20 0.00017  2.04094E+20 0.00012  1.34045E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38899E+20 0.00035  2.38899E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.16509E+22 0.00026  3.00210E+22 0.00023  1.62986E+21 0.00183 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.17652E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39263E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06791E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.68996E-01 0.06991 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.88388E-01 0.00931 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.36729E-04 0.01615 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.79669E+03 0.01906 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08905E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.51512E-01 0.05944 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.10393E-01 0.05944 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52122E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.89134E-01 0.00042  6.16308E-02 0.00041  2.00012E-04 0.00837 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.89321E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.89716E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.89321E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08845E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05253E+00 0.00017 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05262E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29821E-02 0.00123 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29795E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47129E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46955E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.27071E-03 0.00561  2.62818E-04 0.02070  8.12763E-04 0.01171  5.98508E-04 0.01219  1.27675E-03 0.00869  2.66747E-04 0.01871  5.31202E-05 0.04362 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.49693E-01 0.01530  1.24758E-02 3.0E-05  3.23337E-02 8.8E-05  1.06084E-01 0.00058  2.97345E-01 0.00027  1.23269E+00 0.00252  4.63125E+00 0.04037 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23709E-03 0.00833  2.64262E-04 0.03073  8.15006E-04 0.01611  5.85769E-04 0.01908  1.26288E-03 0.01300  2.57192E-04 0.02944  5.19736E-05 0.06487 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44729E-01 0.02220  1.24761E-02 3.3E-05  3.23351E-02 0.00013  1.06153E-01 0.00088  2.97223E-01 0.00037  1.23580E+00 0.00031  6.37922E+00 0.02772 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.47331E-07 0.00114  8.47023E-07 0.00114  9.43295E-07 0.01906 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.38055E-07 0.00104  8.37752E-07 0.00104  9.32669E-07 0.01894 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22877E-03 0.00860  2.55098E-04 0.03259  7.91723E-04 0.01839  5.88399E-04 0.02152  1.27807E-03 0.01396  2.61483E-04 0.03158  5.40007E-05 0.07169 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53594E-01 0.02467  1.24756E-02 5.5E-05  3.23350E-02 0.00016  1.06152E-01 0.00101  2.97532E-01 0.00044  1.23552E+00 0.00039  6.54719E+00 0.03917 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.51565E-07 0.01705  7.50796E-07 0.01705  9.28046E-07 0.07498 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.43462E-07 0.01704  7.42703E-07 0.01704  9.17644E-07 0.07488 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.04194E-03 0.04182  2.82698E-04 0.14341  7.12746E-04 0.07633  5.74044E-04 0.08439  1.20980E-03 0.06110  1.91754E-04 0.14903  7.08983E-05 0.24958 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.84367E-01 0.11658  1.24761E-02 0.00013  3.23343E-02 0.00049  1.06585E-01 0.00346  2.97496E-01 0.00153  1.23918E+00 0.00090  6.34062E+00 0.11079 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00946E-03 0.04082  2.81953E-04 0.13780  7.07287E-04 0.07386  5.67378E-04 0.08076  1.17994E-03 0.06023  1.98528E-04 0.14692  7.43814E-05 0.24649 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.87299E-01 0.11898  1.24761E-02 0.00013  3.23338E-02 0.00048  1.06559E-01 0.00343  2.97486E-01 0.00151  1.23928E+00 0.00089  6.34062E+00 0.11079 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.08035E+03 0.03890 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.42021E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.32809E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28206E-03 0.00579 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.89861E+03 0.00584 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.18911E-09 0.00124 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.16919E-05 0.00438  7.16420E-05 0.00441  2.53159E-06 0.29774 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.82871E-05 0.01668  7.82967E-05 0.01673  2.33318E-06 0.38226 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.21024E-04 0.01554  4.21204E-04 0.01557  3.77779E-04 0.28613 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71983E+01 0.01135 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46529E+01 0.00025  4.05177E+01 0.00032 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 14:13:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01523E+00  1.00294E+00  9.99128E-01  9.99755E-01  1.00377E+00  9.95712E-01  1.00016E+00  9.96300E-01  1.00454E+00  9.96613E-01  9.95607E-01  1.00549E+00  9.96888E-01  9.98410E-01  9.98945E-01  1.00038E+00  9.92648E-01  9.97489E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.10847E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58915E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39620E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53406E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65920E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49111E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48208E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20051E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.75984E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500747 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50030E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50030E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.00111E+03 ;
RUNNING_TIME              (idx, 1)        =  1.01618E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.42733E-01  1.29867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  9.99408E+01  3.86701E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.73500E-02  1.93333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.49817E-01  5.09833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.01609E+02  5.43692E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85166 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98548E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84905E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.38127E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65356E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.23582E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85330E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29539E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49701E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49298E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.74787E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39693E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.85061E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.95980E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89704E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60095E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49556E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34604E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33327E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54287E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.66193E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17891E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.00453E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51093E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.32712E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.49857E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.67673E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.75727E-04 -1.27119E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.95804E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.57724E+18 0.00394  1.68186E-02 0.00392 ];
U233_FISS                 (idx, [1:   4]) = [  9.20822E+19 0.00050  9.81876E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.01792E+16 0.04867  1.08527E-04 0.04868 ];
TH232_CAPT                (idx, [1:   4]) = [  9.42290E+19 0.00053  7.44587E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34592E+19 0.00138  1.06353E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  3.45782E+15 0.08046  2.72920E-05 0.08037 ];
SM149_CAPT                (idx, [1:   4]) = [  6.03739E+16 0.01928  4.77161E-04 0.01930 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001218 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16785E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001218 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5225642 5.23122E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3872611 3.87665E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 902965 9.03810E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001218 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.79397E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36420E+20 3.7E-06  2.36420E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37758E+19 1.2E-07  9.37758E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26607E+20 0.00032  1.12935E+20 0.00023  1.36722E+19 0.00200 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20383E+20 0.00019  2.06710E+20 0.00013  1.36722E+19 0.00200 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41918E+20 0.00036  2.41918E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.21117E+22 0.00027  3.04498E+22 0.00024  1.66186E+21 0.00194 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.18657E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42248E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08549E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51431E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51431E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.35672E-01 0.05492 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.98257E-01 0.01226 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.85823E-04 0.02120 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.44056E+03 0.02418 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09625E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 9.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.36577E-01 0.04152 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.79044E-01 0.04152 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52112E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77283E-01 0.00044  6.08848E-02 0.00043  1.98676E-04 0.00869 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.77083E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.77322E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.77083E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07414E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05953E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05835E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28926E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29055E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47849E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.47845E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.32819E-03 0.00551  2.68819E-04 0.01972  8.16737E-04 0.01098  6.04994E-04 0.01311  1.31542E-03 0.00867  2.73308E-04 0.01886  4.89045E-05 0.04656 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43527E-01 0.01480  1.24761E-02 2.7E-05  3.23360E-02 9.6E-05  1.06121E-01 0.00063  2.97312E-01 0.00026  1.23614E+00 0.00023  4.54306E+00 0.04284 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24885E-03 0.00833  2.59996E-04 0.02815  7.96443E-04 0.01663  5.89356E-04 0.01941  1.28100E-03 0.01349  2.72652E-04 0.02812  4.94076E-05 0.06531 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.50446E-01 0.02175  1.24761E-02 3.4E-05  3.23411E-02 0.00015  1.06043E-01 0.00083  2.97421E-01 0.00038  1.23616E+00 0.00029  6.60763E+00 0.02749 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.61884E-07 0.00131  8.61533E-07 0.00132  9.70831E-07 0.02083 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.42249E-07 0.00126  8.41907E-07 0.00127  9.48777E-07 0.02084 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24880E-03 0.00882  2.67940E-04 0.03112  7.97423E-04 0.01869  5.91293E-04 0.02116  1.27305E-03 0.01425  2.70639E-04 0.03187  4.84555E-05 0.07780 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47334E-01 0.02598  1.24756E-02 5.2E-05  3.23325E-02 0.00016  1.06157E-01 0.00111  2.97309E-01 0.00043  1.23601E+00 0.00038  6.54332E+00 0.04153 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63686E-07 0.01708  7.63418E-07 0.01708  7.89858E-07 0.05619 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.46541E-07 0.01708  7.46278E-07 0.01707  7.72633E-07 0.05624 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83400E-03 0.04211  2.47858E-04 0.12615  6.77908E-04 0.08096  5.64219E-04 0.08603  1.07593E-03 0.06907  2.22019E-04 0.12559  4.60672E-05 0.27943 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.48774E-01 0.08555  1.24776E-02 9.7E-05  3.23442E-02 0.00058  1.06681E-01 0.00357  2.96833E-01 0.00143  1.23753E+00 0.00095  7.25185E+00 0.10579 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.80977E-03 0.04138  2.50919E-04 0.12424  6.74455E-04 0.07915  5.49655E-04 0.08445  1.06324E-03 0.06823  2.24534E-04 0.12846  4.69645E-05 0.26201 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.48958E-01 0.08265  1.24774E-02 0.00010  3.23451E-02 0.00058  1.06682E-01 0.00355  2.96828E-01 0.00141  1.23752E+00 0.00095  7.25185E+00 0.10579 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75784E+03 0.03948 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.54956E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.35472E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25504E-03 0.00593 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.80830E+03 0.00601 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.01547E-09 0.00104 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.95809E-05 0.00631  6.95571E-05 0.00632  1.21643E-06 0.39054 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.96738E-05 0.02184  6.96387E-05 0.02180  1.39910E-06 0.60757 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.78286E-04 0.02045  2.78513E-04 0.02047  1.91592E-04 0.37675 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74000E+01 0.01078 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48208E+01 0.00026  4.06821E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 14:53:10 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00654E+00  9.95769E-01  1.00670E+00  9.99265E-01  1.00062E+00  9.95456E-01  9.97226E-01  9.96573E-01  1.01084E+00  1.00252E+00  9.98527E-01  1.00510E+00  9.96357E-01  9.93803E-01  9.99304E-01  1.00025E+00  9.99958E-01  9.95188E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.09445E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59056E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39747E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53487E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65460E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49747E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48840E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20915E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.75302E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500420 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50023E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50023E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.39276E+03 ;
RUNNING_TIME              (idx, 1)        =  1.40966E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.75183E-01  1.32450E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.39125E+02  3.91842E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.55667E-02  1.82167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.99233E-01  4.93833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.40958E+02  5.67986E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88008 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98545E+00 2.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.87901E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.49876E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66945E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.14993E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89096E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32080E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57650E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50602E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.80262E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.55868E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.00616E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.15419E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79538E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74324E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48841E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36285E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34617E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56023E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39216E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42584E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07782E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54026E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.14785E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.66010E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.77008E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12670E-03 -2.48773E+26  2.21046E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.10315E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58963E+18 0.00378  1.69395E-02 0.00366 ];
U233_FISS                 (idx, [1:   4]) = [  9.19817E+19 0.00051  9.80298E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.30717E+16 0.02523  4.59193E-04 0.02525 ];
TH232_CAPT                (idx, [1:   4]) = [  9.53367E+19 0.00052  7.41788E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34112E+19 0.00139  1.04348E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  1.40752E+16 0.04162  1.09504E-04 0.04160 ];
SM149_CAPT                (idx, [1:   4]) = [  9.95786E+16 0.01591  7.74884E-04 0.01592 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000920 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15233E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000920 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5256275 5.26194E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3837584 3.84159E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907061 9.08002E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000920 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -6.33299E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36414E+20 3.5E-06  2.36414E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 1.2E-07  9.37751E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28458E+20 0.00032  1.14645E+20 0.00023  1.38124E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22233E+20 0.00019  2.08420E+20 0.00013  1.38124E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.44252E+20 0.00035  2.44252E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24304E+22 0.00026  3.07553E+22 0.00023  1.67512E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.21793E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44412E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09748E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51440E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51440E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.11863E-01 0.05018 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.15017E-01 0.01181 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.22060E-04 0.02244 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.32293E+03 0.02377 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09203E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.63192E-01 0.03899 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.03106E-01 0.03899 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52108E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.68304E-01 0.00045  6.03344E-02 0.00043  1.97044E-04 0.00895 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.68410E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.67960E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.68410E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06518E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05906E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05932E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28987E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28928E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49519E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49056E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39321E-03 0.00521  2.72474E-04 0.01898  8.33352E-04 0.01195  6.14540E-04 0.01240  1.34172E-03 0.00865  2.78686E-04 0.01841  5.24334E-05 0.04315 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43453E-01 0.01420  1.24750E-02 3.0E-05  3.23314E-02 9.4E-05  1.06159E-01 0.00061  2.97422E-01 0.00026  1.23263E+00 0.00252  4.51211E+00 0.04100 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.30665E-03 0.00780  2.68415E-04 0.03032  8.08000E-04 0.01751  6.03803E-04 0.01931  1.30475E-03 0.01252  2.69153E-04 0.02874  5.25285E-05 0.07117 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48124E-01 0.02423  1.24751E-02 3.6E-05  3.23320E-02 0.00014  1.06018E-01 0.00079  2.97455E-01 0.00038  1.23626E+00 0.00031  6.29467E+00 0.02859 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68201E-07 0.00126  8.67864E-07 0.00126  9.69478E-07 0.01766 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.40620E-07 0.00119  8.40294E-07 0.00119  9.38650E-07 0.01764 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.25484E-03 0.00915  2.63774E-04 0.03245  7.86316E-04 0.01942  5.96828E-04 0.01978  1.29711E-03 0.01427  2.61124E-04 0.03190  4.96938E-05 0.07234 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.48936E-01 0.02507  1.24749E-02 5.6E-05  3.23378E-02 0.00020  1.06074E-01 0.00108  2.97588E-01 0.00043  1.23486E+00 0.00043  6.51238E+00 0.04024 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63852E-07 0.01714  7.63643E-07 0.01714  7.86774E-07 0.04993 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.39980E-07 0.01714  7.39779E-07 0.01714  7.61919E-07 0.04988 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.10531E-03 0.04096  2.65186E-04 0.12291  6.61071E-04 0.08017  5.90572E-04 0.08208  1.29950E-03 0.06164  2.29832E-04 0.14258  5.91524E-05 0.31410 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.38158E-01 0.07881  1.24756E-02 0.00013  3.23681E-02 0.00064  1.06134E-01 0.00301  2.98024E-01 0.00164  1.23650E+00 0.00102  5.72651E+00 0.11623 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10959E-03 0.04063  2.59108E-04 0.12147  6.72184E-04 0.07793  5.85210E-04 0.08062  1.29663E-03 0.06157  2.39001E-04 0.14147  5.74534E-05 0.30979 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.42878E-01 0.07577  1.24756E-02 0.00013  3.23669E-02 0.00063  1.06111E-01 0.00298  2.98013E-01 0.00163  1.23649E+00 0.00102  5.75823E+00 0.11675 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.09994E+03 0.03781 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60930E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33576E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.36302E-03 0.00620 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90721E+03 0.00625 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93617E-09 0.00086 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77815E-05 0.00763  6.77722E-05 0.00766  7.43234E-07 0.51686 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.29918E-05 0.02536  6.30774E-05 0.02537  3.06784E-07 0.57541 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.14203E-04 0.02175  2.14532E-04 0.02179  1.21211E-04 0.50058 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71276E+01 0.01111 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48840E+01 0.00028  4.07458E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 15:33:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01983E+00  1.00295E+00  1.00370E+00  9.96993E-01  9.92322E-01  1.00246E+00  9.99071E-01  1.00526E+00  1.00218E+00  1.00086E+00  9.97523E-01  1.00391E+00  9.91035E-01  9.93622E-01  9.91486E-01  9.95621E-01  1.00291E+00  9.98254E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.08465E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59154E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39540E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53264E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65180E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51340E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50434E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24637E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.75479E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500347 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50029E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50029E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.79677E+03 ;
RUNNING_TIME              (idx, 1)        =  1.81560E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.13517E-01  1.38333E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.79543E+02  4.04175E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.51833E-02  1.96167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.50000E-01  5.07167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.81552E+02  5.74974E+02 ];
CPU_USAGE                 (idx, 1)        = 9.89630 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98577E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89631E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.52585E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66437E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.52691E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91276E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33439E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58149E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49928E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.98288E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.67942E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.27714E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.25262E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.70005E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85405E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93714E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32742E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31904E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52287E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34230E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81499E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07546E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51426E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.50469E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70631E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.89947E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.28583E-03 -5.04704E+26  2.21302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.33588E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.60185E+18 0.00418  1.70642E-02 0.00412 ];
U233_FISS                 (idx, [1:   4]) = [  9.16907E+19 0.00048  9.76828E-01 8.4E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.58631E+17 0.01264  1.68982E-03 0.01262 ];
TH232_CAPT                (idx, [1:   4]) = [  9.70312E+19 0.00053  7.38056E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33927E+19 0.00126  1.01872E-01 0.00121 ];
U235_CAPT                 (idx, [1:   4]) = [  5.72087E+16 0.02097  4.35080E-04 0.02095 ];
SM149_CAPT                (idx, [1:   4]) = [  1.39960E+17 0.01296  1.06468E-03 0.01297 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001147 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13802E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001147 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5306502 5.31217E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3789042 3.79281E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905603 9.06397E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001147 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.91852E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36401E+20 3.7E-06  2.36401E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.3E-07  9.37729E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31495E+20 0.00030  1.17464E+20 0.00022  1.40302E+19 0.00187 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.25267E+20 0.00017  2.11237E+20 0.00012  1.40302E+19 0.00187 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47487E+20 0.00035  2.47487E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29440E+22 0.00025  3.12517E+22 0.00023  1.69228E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24333E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47701E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11596E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51186E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51186E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.76823E-01 0.05079 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07894E-01 0.01318 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93733E-04 0.02348 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.38504E+03 0.02350 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09364E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.48788E-01 0.03940 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.90038E-01 0.03940 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52100E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.56087E-01 0.00041  5.95703E-02 0.00040  1.90076E-04 0.00886 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55499E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.55255E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55499E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05077E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06035E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05982E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28820E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28864E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51392E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51292E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39715E-03 0.00584  2.67995E-04 0.02108  8.29629E-04 0.01146  6.30989E-04 0.01327  1.33292E-03 0.00879  2.78100E-04 0.01950  5.75221E-05 0.04239 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.52989E-01 0.01468  1.24131E-02 0.00355  3.23314E-02 9.8E-05  1.06130E-01 0.00059  2.97628E-01 0.00027  1.23340E+00 0.00252  4.73082E+00 0.03893 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22344E-03 0.00798  2.64324E-04 0.03077  7.91016E-04 0.01680  6.06742E-04 0.01879  1.24745E-03 0.01287  2.63510E-04 0.02774  5.03896E-05 0.05967 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42621E-01 0.02032  1.24754E-02 4.0E-05  3.23302E-02 0.00014  1.06049E-01 0.00075  2.97490E-01 0.00037  1.23682E+00 0.00040  6.15295E+00 0.02813 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82661E-07 0.00128  8.82344E-07 0.00128  9.83791E-07 0.01606 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.43852E-07 0.00124  8.43549E-07 0.00124  9.40522E-07 0.01605 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.17706E-03 0.00905  2.55607E-04 0.03470  7.73115E-04 0.01924  5.75673E-04 0.02094  1.25128E-03 0.01428  2.63793E-04 0.03168  5.75936E-05 0.07053 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.59255E-01 0.02537  1.24744E-02 5.8E-05  3.23331E-02 0.00017  1.06107E-01 0.00099  2.97518E-01 0.00043  1.23704E+00 0.00055  6.05766E+00 0.04098 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.78616E-07 0.01718  7.78406E-07 0.01718  8.63284E-07 0.06288 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.44482E-07 0.01717  7.44280E-07 0.01717  8.25165E-07 0.06273 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.81987E-03 0.04352  1.76463E-04 0.14936  7.28978E-04 0.08465  5.38015E-04 0.08909  1.11325E-03 0.06405  2.27029E-04 0.13703  3.61393E-05 0.26908 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.21956E-01 0.10544  1.24738E-02 0.00019  3.23447E-02 0.00060  1.05892E-01 0.00303  2.97833E-01 0.00173  1.23551E+00 0.00115  5.81117E+00 0.12524 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.78905E-03 0.04320  1.80386E-04 0.14265  7.18849E-04 0.08416  5.25388E-04 0.08705  1.10101E-03 0.06343  2.26415E-04 0.13390  3.70090E-05 0.27205 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.22513E-01 0.10593  1.24738E-02 0.00019  3.23485E-02 0.00062  1.05895E-01 0.00303  2.97908E-01 0.00173  1.23551E+00 0.00115  5.81117E+00 0.12524 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65039E+03 0.04035 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.75007E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.36527E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22563E-03 0.00545 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.68759E+03 0.00556 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92407E-09 0.00084 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84037E-05 0.00771  6.81179E-05 0.00815  1.42394E-06 0.38105 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.86486E-05 0.02373  5.85990E-05 0.02397  6.48664E-07 0.54072 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.88247E-04 0.02231  1.88194E-04 0.02251  2.16659E-04 0.37657 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70034E+01 0.01178 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50434E+01 0.00027  4.08566E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 16:14:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01959E+00  9.98875E-01  9.98463E-01  9.99646E-01  9.97725E-01  9.98862E-01  1.00486E+00  1.00227E+00  1.00205E+00  9.99182E-01  9.97346E-01  9.94974E-01  9.95674E-01  1.00575E+00  9.97013E-01  9.98980E-01  9.96438E-01  9.92309E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.08286E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59171E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39382E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53105E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64812E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52540E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51632E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27405E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.76167E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500602 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50029E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50029E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.20407E+03 ;
RUNNING_TIME              (idx, 1)        =  2.22575E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.54550E-01  1.41033E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.20310E+02  4.07671E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.52667E-02  2.00833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.23533E-01  7.34833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.22543E+02  5.87751E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90259 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98578E+00 2.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90325E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.59605E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67420E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.31535E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94956E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35875E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61447E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50628E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09597E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79078E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.68874E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.42092E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25627E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94840E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35271E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33802E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32682E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53344E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.01290E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17372E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11645E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54702E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.32804E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.81151E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.99780E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.42740E-03 -7.56761E+26  2.21554E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.52986E-01 0.00064 ];
TH232_FISS                (idx, [1:   4]) = [  1.61273E+18 0.00404  1.72014E-02 0.00400 ];
U233_FISS                 (idx, [1:   4]) = [  9.11950E+19 0.00052  9.72692E-01 9.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.42403E+17 0.00830  3.65237E-03 0.00830 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81819E+19 0.00050  7.33990E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33426E+19 0.00137  9.97463E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  1.21770E+17 0.01406  9.10512E-04 0.01411 ];
SM149_CAPT                (idx, [1:   4]) = [  1.55481E+17 0.01272  1.16229E-03 0.01272 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001177 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16119E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001177 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5346128 5.35181E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3747213 3.75106E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907836 9.08743E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001177 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.45058E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36385E+20 3.7E-06  2.36385E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37700E+19 1.2E-07  9.37700E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33755E+20 0.00028  1.19565E+20 0.00020  1.41898E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27525E+20 0.00016  2.13335E+20 0.00011  1.41898E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49945E+20 0.00036  2.49945E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.33414E+22 0.00024  3.16282E+22 0.00022  1.71313E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.27147E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.50240E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.13006E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50803E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50803E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.94851E-01 0.04967 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.46605E-01 0.01062 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.95602E-04 0.02272 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.15702E+03 0.02366 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09129E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.68171E-01 0.03735 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.07470E-01 0.03735 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52090E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.45589E-01 0.00042  5.89101E-02 0.00041  1.88995E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.45740E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.45797E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.45740E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04027E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06238E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06087E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28554E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28729E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53038E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53351E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.45761E-03 0.00571  2.85609E-04 0.01985  8.39500E-04 0.01149  6.40705E-04 0.01322  1.35156E-03 0.00921  2.85850E-04 0.01932  5.43849E-05 0.04542 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47598E-01 0.01546  1.24760E-02 2.8E-05  3.23223E-02 9.8E-05  1.05991E-01 0.00053  2.97612E-01 0.00029  1.23736E+00 0.00030  4.64375E+00 0.04047 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26510E-03 0.00805  2.75436E-04 0.02956  7.98884E-04 0.01614  5.96536E-04 0.01962  1.27516E-03 0.01266  2.68229E-04 0.02885  5.08535E-05 0.06462 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42355E-01 0.02099  1.24764E-02 3.1E-05  3.23214E-02 0.00012  1.05933E-01 0.00075  2.97753E-01 0.00043  1.23735E+00 0.00039  6.44604E+00 0.02797 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.96462E-07 0.00122  8.96207E-07 0.00122  9.68001E-07 0.02279 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.47628E-07 0.00115  8.47386E-07 0.00115  9.15436E-07 0.02281 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19242E-03 0.00925  2.63216E-04 0.03407  7.78067E-04 0.01765  6.06382E-04 0.02125  1.23373E-03 0.01503  2.61418E-04 0.03220  4.96103E-05 0.07203 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.46338E-01 0.02476  1.24762E-02 4.7E-05  3.23230E-02 0.00017  1.05913E-01 0.00092  2.97697E-01 0.00048  1.23746E+00 0.00056  6.57819E+00 0.04055 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.94731E-07 0.01708  7.94471E-07 0.01709  8.38284E-07 0.05306 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.51542E-07 0.01708  7.51298E-07 0.01708  7.92353E-07 0.05300 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93522E-03 0.04227  2.13341E-04 0.17316  7.88984E-04 0.07853  5.93538E-04 0.09348  1.01542E-03 0.06257  2.85962E-04 0.11836  3.79765E-05 0.31489 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.64237E-01 0.09758  1.24754E-02 0.00016  3.22919E-02 0.00039  1.05975E-01 0.00293  2.98171E-01 0.00174  1.23897E+00 0.00171  5.94420E+00 0.12713 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.88288E-03 0.04189  2.13726E-04 0.16995  7.61001E-04 0.07764  5.88434E-04 0.08953  9.89348E-04 0.06205  2.95412E-04 0.11824  3.49599E-05 0.30440 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.59869E-01 0.09280  1.24754E-02 0.00016  3.22918E-02 0.00038  1.05980E-01 0.00293  2.98201E-01 0.00173  1.23909E+00 0.00170  5.89352E+00 0.12798 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.72493E+03 0.03943 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.90388E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.41879E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25235E-03 0.00566 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.65339E+03 0.00569 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93817E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72975E-05 0.00754  6.72437E-05 0.00755  1.42228E-06 0.36073 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.05640E-05 0.02618  6.09693E-05 0.02711  8.65561E-07 0.45015 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.89327E-04 0.02199  1.89180E-04 0.02205  2.31153E-04 0.35288 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70290E+01 0.01164 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51632E+01 0.00028  4.09206E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 16:56:29 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01345E+00  9.95713E-01  9.94589E-01  1.00638E+00  9.94543E-01  1.00437E+00  1.00038E+00  9.99953E-01  9.99751E-01  1.00010E+00  9.98849E-01  9.99894E-01  1.00470E+00  1.00155E+00  9.95373E-01  9.96765E-01  9.94027E-01  9.99620E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.07984E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59202E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39009E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52731E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64875E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53796E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52888E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.31067E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.76974E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500073 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50053E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50053E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.61919E+03 ;
RUNNING_TIME              (idx, 1)        =  2.64287E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.98067E-01  1.43517E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.61836E+02  4.15262E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.13600E-01  1.83333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.73050E-01  4.94667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.64279E+02  5.91710E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91040 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98486E+00 2.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91165E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.62200E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67200E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.35433E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97512E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37384E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61452E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50226E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26735E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.92376E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.68790E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.56667E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93997E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06619E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07121E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.31039E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30466E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50352E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.82743E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78742E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.12007E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55515E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.90359E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85335E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.01460E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43284E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.80072E-03 -1.28078E+27  2.22078E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.82681E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.63825E+18 0.00392  1.74730E-02 0.00389 ];
U233_FISS                 (idx, [1:   4]) = [  9.03259E+19 0.00054  9.63368E-01 9.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.36390E+17 0.00513  8.92069E-03 0.00511 ];
PU239_FISS                (idx, [1:   4]) = [  2.60678E+13 1.00000  2.74937E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  9.98683E+19 0.00053  7.28313E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32210E+19 0.00133  9.64191E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  2.95316E+17 0.00986  2.15370E-03 0.00986 ];
SM149_CAPT                (idx, [1:   4]) = [  1.64810E+17 0.01147  1.20215E-03 0.01150 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002111 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15821E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002111 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5400850 5.40603E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3693016 3.69648E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908245 9.09065E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002111 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.11759E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36350E+20 3.9E-06  2.36350E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37626E+19 1.2E-07  9.37626E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37055E+20 0.00030  1.22647E+20 0.00022  1.44077E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30818E+20 0.00018  2.16410E+20 0.00012  1.44077E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53650E+20 0.00037  2.53650E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.39281E+22 0.00026  3.21900E+22 0.00024  1.73809E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.30592E+19 0.00150 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53877E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15001E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50046E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50046E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.52318E-01 0.05142 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.14311E-01 0.01294 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93356E-04 0.02373 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.35723E+03 0.02222 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09096E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.30449E-01 0.03961 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.73211E-01 0.03961 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52073E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99702E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.31848E-01 0.00044  5.80469E-02 0.00043  1.88513E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.32051E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.31848E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.32051E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02528E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05513E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.05494E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.29495E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.29493E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.57936E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.57802E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.55705E-03 0.00514  2.80989E-04 0.01907  8.62901E-04 0.01149  6.47261E-04 0.01287  1.41307E-03 0.00858  2.95095E-04 0.01920  5.77301E-05 0.04606 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53015E-01 0.01551  1.24452E-02 0.00251  3.23152E-02 0.00011  1.06305E-01 0.00059  2.98085E-01 0.00026  1.23842E+00 0.00038  4.66963E+00 0.04041 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29260E-03 0.00758  2.56267E-04 0.02815  7.86582E-04 0.01592  6.20530E-04 0.01878  1.30545E-03 0.01315  2.68930E-04 0.02830  5.48476E-05 0.06579 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56071E-01 0.02403  1.24763E-02 3.4E-05  3.23087E-02 0.00014  1.06307E-01 0.00082  2.98148E-01 0.00040  1.23864E+00 0.00049  6.51129E+00 0.02716 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.08530E-07 0.00122  9.08172E-07 0.00121  1.02159E-06 0.01870 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.46545E-07 0.00113  8.46212E-07 0.00113  9.51800E-07 0.01865 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22316E-03 0.00916  2.48185E-04 0.03556  7.81573E-04 0.01851  5.95260E-04 0.02200  1.27839E-03 0.01465  2.73638E-04 0.03070  4.61090E-05 0.07783 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47215E-01 0.02584  1.24761E-02 5.0E-05  3.23098E-02 0.00019  1.06159E-01 0.00099  2.98012E-01 0.00047  1.23770E+00 0.00055  6.33393E+00 0.04368 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.04384E-07 0.01715  8.04188E-07 0.01716  8.32143E-07 0.05052 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.49645E-07 0.01715  7.49462E-07 0.01715  7.75486E-07 0.05050 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.86353E-03 0.04303  2.07817E-04 0.17202  6.52615E-04 0.08490  5.92291E-04 0.08967  1.11466E-03 0.06836  2.62333E-04 0.12957  3.38153E-05 0.32544 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.35965E-01 0.10901  1.24758E-02 0.00016  3.22935E-02 0.00053  1.05579E-01 0.00252  2.98106E-01 0.00170  1.23471E+00 0.00179  5.61932E+00 0.17678 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89553E-03 0.04230  2.10181E-04 0.16797  6.72373E-04 0.08417  5.91895E-04 0.08913  1.12493E-03 0.06670  2.60484E-04 0.12908  3.56682E-05 0.32984 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.34606E-01 0.11172  1.24758E-02 0.00016  3.22924E-02 0.00052  1.05574E-01 0.00251  2.98117E-01 0.00171  1.23508E+00 0.00190  5.61932E+00 0.17678 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.59204E+03 0.04009 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.02630E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.41046E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26540E-03 0.00552 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.61855E+03 0.00559 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95491E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71060E-05 0.00948  6.71024E-05 0.00949  9.25134E-07 0.45073 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.90399E-05 0.02847  5.91662E-05 0.02847  3.85227E-07 0.73335 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84794E-04 0.02367  1.84948E-04 0.02367  1.47472E-04 0.44585 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68133E+01 0.01140 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52888E+01 0.00026  4.09136E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 17:38:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01585E+00  1.00089E+00  9.93649E-01  1.00023E+00  9.96615E-01  1.00317E+00  9.97000E-01  1.00012E+00  1.00206E+00  9.94008E-01  9.97954E-01  9.95321E-01  9.96837E-01  1.00381E+00  9.98346E-01  1.00170E+00  1.00253E+00  9.99888E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.08455E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59155E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38779E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52520E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64723E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54260E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53351E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.32689E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.77786E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500225 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50035E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50035E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.03640E+03 ;
RUNNING_TIME              (idx, 1)        =  3.06218E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.14348E+00  1.45417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.03576E+02  4.17403E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.32767E-01  1.91667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.23233E-01  5.01333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.06211E+02  5.98046E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91579 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98455E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91773E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.70127E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68514E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.05237E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00498E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38910E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.66322E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51322E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75219E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22479E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.12762E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.82305E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23206E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33928E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32967E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30641E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29825E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49653E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.54750E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27981E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.15763E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.65994E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84683E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94074E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03178E+16 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.18164E-02 -2.60904E+27  2.23406E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01674E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.62945E+18 0.00402  1.73786E-02 0.00392 ];
U233_FISS                 (idx, [1:   4]) = [  8.80702E+19 0.00055  9.39389E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.34801E+18 0.00321  2.50454E-02 0.00319 ];
PU239_FISS                (idx, [1:   4]) = [  1.05989E+15 0.15982  1.13265E-05 0.15996 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00757E+20 0.00053  7.14543E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28947E+19 0.00136  9.14491E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  8.23624E+17 0.00580  5.84130E-03 0.00582 ];
U238_CAPT                 (idx, [1:   4]) = [  2.58108E+13 1.00000  1.85984E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  3.88233E+14 0.25365  2.74677E-06 0.25365 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75315E+17 0.01238  1.24332E-03 0.01238 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001418 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12928E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001418 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5461036 5.46666E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3631171 3.63462E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909211 9.10015E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001418 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.68221E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36255E+20 4.1E-06  2.36255E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37405E+19 1.1E-07  9.37405E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40941E+20 0.00029  1.26217E+20 0.00020  1.47236E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34681E+20 0.00017  2.19958E+20 0.00012  1.47236E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57945E+20 0.00038  2.57945E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45464E+22 0.00026  3.27726E+22 0.00023  1.77375E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.34749E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58156E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17066E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.48117E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.48117E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.92196E-01 0.05808 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.41934E-01 0.01171 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.83900E-04 0.02471 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.37744E+03 0.02241 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09001E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.59288E-01 0.04483 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.08452E-01 0.04483 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52030E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99748E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.15962E-01 0.00043  5.70621E-02 0.00043  1.89460E-04 0.00947 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.16223E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.15960E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.16223E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00798E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02931E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02569E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32884E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33338E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.67181E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.67733E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.64392E-03 0.00541  2.80372E-04 0.01831  8.82879E-04 0.01122  6.60519E-04 0.01276  1.44484E-03 0.00827  3.10449E-04 0.01973  6.48566E-05 0.04161 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.64572E-01 0.01467  1.24443E-02 0.00251  3.23017E-02 0.00012  1.06309E-01 0.00059  2.98842E-01 0.00029  1.24284E+00 0.00257  4.95175E+00 0.03611 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28637E-03 0.00855  2.51020E-04 0.02667  8.00863E-04 0.01647  6.18289E-04 0.01845  1.28674E-03 0.01279  2.70459E-04 0.02886  5.90039E-05 0.06207 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.64316E-01 0.02358  1.24757E-02 3.6E-05  3.23018E-02 0.00016  1.06359E-01 0.00083  2.98768E-01 0.00042  1.24732E+00 0.00079  6.27347E+00 0.02652 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.16910E-07 0.00126  9.16566E-07 0.00126  1.03192E-06 0.02150 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39781E-07 0.00116  8.39466E-07 0.00116  9.45178E-07 0.02144 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.31293E-03 0.00950  2.53762E-04 0.03275  8.20815E-04 0.01813  6.05308E-04 0.02194  1.29763E-03 0.01532  2.80177E-04 0.03030  5.52341E-05 0.07322 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.54067E-01 0.02333  1.24749E-02 5.8E-05  3.22917E-02 0.00020  1.06212E-01 0.00104  2.98864E-01 0.00049  1.24601E+00 0.00098  6.22836E+00 0.03954 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.09519E-07 0.01698  8.09397E-07 0.01698  8.20511E-07 0.05608 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.41634E-07 0.01698  7.41521E-07 0.01698  7.51698E-07 0.05607 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98216E-03 0.04311  2.35209E-04 0.13926  6.84198E-04 0.08551  5.81683E-04 0.08780  1.20622E-03 0.06510  2.38615E-04 0.13161  3.62416E-05 0.27907 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.49716E-01 0.09451  1.24732E-02 0.00018  3.23007E-02 0.00060  1.06212E-01 0.00304  2.98493E-01 0.00170  1.24403E+00 0.00257  7.42985E+00 0.10185 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97071E-03 0.04217  2.32458E-04 0.13815  6.85059E-04 0.08392  5.74415E-04 0.08517  1.19921E-03 0.06324  2.42397E-04 0.13437  3.71749E-05 0.26886 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52464E-01 0.09607  1.24734E-02 0.00018  3.23003E-02 0.00061  1.06193E-01 0.00303  2.98566E-01 0.00171  1.24400E+00 0.00257  7.42985E+00 0.10185 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.70218E+03 0.03980 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.09662E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.33151E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34109E-03 0.00556 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.67416E+03 0.00567 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95566E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63760E-05 0.01015  6.64129E-05 0.01015  7.94446E-07 0.41936 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.68724E-05 0.02636  5.68798E-05 0.02644  8.81959E-07 0.49892 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.77471E-04 0.02411  1.77514E-04 0.02406  1.62269E-04 0.40910 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67350E+01 0.01020 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53351E+01 0.00026  4.07416E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 18:20:20 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01451E+00  1.00485E+00  9.96440E-01  9.99322E-01  9.96800E-01  9.98159E-01  9.97630E-01  9.99786E-01  1.00031E+00  9.94618E-01  1.00174E+00  1.00151E+00  9.97172E-01  1.00579E+00  9.95741E-01  9.93004E-01  1.00275E+00  9.99877E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.09941E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59006E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38648E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52429E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64635E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52447E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51537E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.29684E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.78052E+00 0.00043  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2499989 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50031E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50031E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.45346E+03 ;
RUNNING_TIME              (idx, 1)        =  3.48134E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.29257E+00  1.49083E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.45295E+02  4.17183E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.52083E-01  1.93167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.74683E-01  5.14000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.48127E+02  5.99579E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91990 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98599E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92221E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.87162E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73056E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59806E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01638E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39122E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82137E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55770E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.13619E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77439E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.05510E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.19676E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04597E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84779E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38250E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38009E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34875E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56896E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.83210E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30609E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26108E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91819E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.43851E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09625E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03854E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.23712E-02 -4.93949E+27  2.25737E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02886E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.60315E+18 0.00409  1.71126E-02 0.00406 ];
U233_FISS                 (idx, [1:   4]) = [  8.46119E+19 0.00055  9.03171E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.75076E+18 0.00230  5.07125E-02 0.00227 ];
U238_FISS                 (idx, [1:   4]) = [  2.59888E+13 1.00000  2.79705E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  2.13897E+16 0.03350  2.28302E-04 0.03349 ];
PU240_FISS                (idx, [1:   4]) = [  8.07038E+14 0.17860  8.61131E-06 0.17872 ];
PU241_FISS                (idx, [1:   4]) = [  5.98174E+14 0.21188  6.38667E-06 0.21166 ];
TH232_CAPT                (idx, [1:   4]) = [  9.87867E+19 0.00053  6.92733E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23659E+19 0.00139  8.67158E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.67345E+18 0.00384  1.17355E-02 0.00384 ];
U238_CAPT                 (idx, [1:   4]) = [  1.55851E+14 0.40577  1.09528E-06 0.40580 ];
PU239_CAPT                (idx, [1:   4]) = [  1.00091E+16 0.04934  7.01886E-05 0.04927 ];
PU240_CAPT                (idx, [1:   4]) = [  1.66658E+15 0.12098  1.16855E-05 0.12094 ];
PU241_CAPT                (idx, [1:   4]) = [  5.32601E+13 0.70622  3.68688E-07 0.70626 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78073E+17 0.01206  1.24874E-03 0.01205 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001250 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12252E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001250 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486793 5.49250E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3604821 3.60826E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909636 9.10463E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001250 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.81842E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36128E+20 4.3E-06  2.36128E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37021E+19 1.1E-07  9.37021E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42603E+20 0.00028  1.27810E+20 0.00020  1.47930E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36305E+20 0.00017  2.21512E+20 0.00011  1.47930E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59636E+20 0.00034  2.59636E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46821E+22 0.00024  3.29057E+22 0.00022  1.77639E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36399E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59945E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17359E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42909E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42909E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.66993E-01 0.05296 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.36538E-01 0.01241 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.86046E-04 0.02479 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.21124E+03 0.02203 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08956E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.97902E-01 0.04109 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.43438E-01 0.04109 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51998E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99830E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09150E-01 0.00043  5.66329E-02 0.00042  1.95322E-04 0.00919 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09409E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09499E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09409E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00049E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.96818E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  6.96671E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.41251E-02 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  1.41437E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.80972E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.81320E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85416E-03 0.00544  2.78893E-04 0.01918  9.15927E-04 0.01099  7.13206E-04 0.01214  1.53060E-03 0.00830  3.42274E-04 0.01860  7.32584E-05 0.03802 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.90024E-01 0.01446  1.24457E-02 0.00251  3.22766E-02 0.00012  1.06447E-01 0.00056  3.00125E-01 0.00030  1.25429E+00 0.00068  5.76890E+00 0.03066 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43829E-03 0.00808  2.45579E-04 0.02922  8.12401E-04 0.01644  6.41390E-04 0.01833  1.36455E-03 0.01273  3.06201E-04 0.02839  6.81703E-05 0.05475 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.95917E-01 0.02109  1.24768E-02 5.2E-05  3.22749E-02 0.00017  1.06516E-01 0.00074  3.00048E-01 0.00042  1.25505E+00 0.00094  7.02595E+00 0.02153 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.00523E-07 0.00128  9.00113E-07 0.00128  1.02136E-06 0.02069 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.18630E-07 0.00116  8.18257E-07 0.00116  9.28506E-07 0.02068 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43849E-03 0.00921  2.42457E-04 0.03375  7.99991E-04 0.01854  6.46101E-04 0.02116  1.36669E-03 0.01412  3.14006E-04 0.03029  6.92428E-05 0.06215 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.99113E-01 0.02441  1.24768E-02 5.3E-05  3.22741E-02 0.00023  1.06588E-01 0.00106  3.00306E-01 0.00052  1.25472E+00 0.00120  6.91785E+00 0.03095 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.93820E-07 0.01715  7.93410E-07 0.01715  8.39397E-07 0.06142 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21775E-07 0.01714  7.21401E-07 0.01714  7.63175E-07 0.06135 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87500E-03 0.04566  1.77187E-04 0.17408  7.32520E-04 0.08607  4.63893E-04 0.10989  1.20179E-03 0.06705  2.48149E-04 0.13889  5.14613E-05 0.25883 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.00109E-01 0.10285  1.24761E-02 0.00017  3.22984E-02 0.00071  1.06645E-01 0.00358  3.00685E-01 0.00191  1.25219E+00 0.00346  6.95413E+00 0.09755 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87121E-03 0.04485  1.72350E-04 0.17059  7.29373E-04 0.08546  4.46425E-04 0.11129  1.22073E-03 0.06536  2.48097E-04 0.13593  5.42377E-05 0.26359 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.05142E-01 0.10188  1.24761E-02 0.00017  3.22981E-02 0.00071  1.06676E-01 0.00362  3.00750E-01 0.00191  1.25235E+00 0.00347  6.95413E+00 0.09755 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65104E+03 0.04319 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.93206E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11999E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.36552E-03 0.00651 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77015E+03 0.00671 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92606E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75308E-05 0.00954  6.75158E-05 0.00955  1.63495E-06 0.33985 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65842E-05 0.02807  5.65351E-05 0.02809  1.42654E-06 0.35181 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79924E-04 0.02428  1.79726E-04 0.02427  2.30477E-04 0.33210 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61773E+01 0.01131 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51537E+01 0.00026  4.01840E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 19:02:40 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01755E+00  1.00405E+00  9.98828E-01  1.00304E+00  9.91472E-01  9.96718E-01  9.96123E-01  1.00215E+00  9.98723E-01  1.00443E+00  9.99991E-01  9.94895E-01  9.99618E-01  1.00584E+00  9.89564E-01  9.89923E-01  1.00227E+00  1.00481E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.10705E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58930E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38592E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52394E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64798E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51325E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50413E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27747E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.78056E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500321 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50015E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50015E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.87465E+03 ;
RUNNING_TIME              (idx, 1)        =  3.90468E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.44702E+00  1.54450E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.87422E+02  4.21270E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.71150E-01  1.90667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.28450E-01  5.37167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.90461E+02  5.99920E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92308 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98625E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92569E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.90385E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73876E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.71737E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01271E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38812E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85705E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56602E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.08925E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99027E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.91406E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.39126E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13518E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04325E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81906E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38654E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35228E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57453E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88178E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73626E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27687E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97570E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.87618E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11925E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03831E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.74655E-02 -6.06430E+27  2.26862E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02676E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59071E+18 0.00409  1.69934E-02 0.00405 ];
U233_FISS                 (idx, [1:   4]) = [  8.34990E+19 0.00053  8.92028E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.41093E+18 0.00231  5.78050E-02 0.00225 ];
PU239_FISS                (idx, [1:   4]) = [  4.98382E+16 0.02253  5.32361E-04 0.02253 ];
PU240_FISS                (idx, [1:   4]) = [  2.12706E+15 0.10583  2.27136E-05 0.10581 ];
PU241_FISS                (idx, [1:   4]) = [  1.81349E+15 0.11581  1.93671E-05 0.11585 ];
TH232_CAPT                (idx, [1:   4]) = [  9.75406E+19 0.00056  6.84191E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22254E+19 0.00150  8.57534E-02 0.00141 ];
U235_CAPT                 (idx, [1:   4]) = [  1.91485E+18 0.00349  1.34321E-02 0.00348 ];
U238_CAPT                 (idx, [1:   4]) = [  1.55750E+14 0.40569  1.09382E-06 0.40569 ];
PU239_CAPT                (idx, [1:   4]) = [  2.26598E+16 0.03362  1.58889E-04 0.03361 ];
PU240_CAPT                (idx, [1:   4]) = [  4.87057E+15 0.07381  3.41637E-05 0.07385 ];
PU241_CAPT                (idx, [1:   4]) = [  2.07061E+14 0.35047  1.45403E-06 0.35050 ];
SM149_CAPT                (idx, [1:   4]) = [  1.76457E+17 0.01194  1.23768E-03 0.01192 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000585 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10838E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000585 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486309 5.49214E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3602400 3.60613E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911876 9.12815E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000585 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.86265E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36118E+20 4.7E-06  2.36118E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36879E+19 1.2E-07  9.36879E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42564E+20 0.00027  1.27830E+20 0.00020  1.47342E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36252E+20 0.00016  2.21518E+20 0.00011  1.47342E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59578E+20 0.00037  2.59578E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46230E+22 0.00026  3.28522E+22 0.00023  1.77076E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36959E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59948E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17040E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.40112E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.40112E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.51078E-01 0.05056 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24675E-01 0.01265 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94953E-04 0.02316 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.14411E+03 0.02219 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08722E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.22842E-01 0.03899 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.66142E-01 0.03899 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52026E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99861E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.08860E-01 0.00044  5.66079E-02 0.00044  1.94647E-04 0.00915 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09358E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09671E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09358E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00067E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94601E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94559E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44432E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44460E-02 0.00102 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86286E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86149E-01 0.00052 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85729E-03 0.00553  2.80973E-04 0.01958  9.05011E-04 0.01102  6.95015E-04 0.01242  1.54917E-03 0.00876  3.52975E-04 0.01807  7.41436E-05 0.03943 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.93299E-01 0.01466  1.24456E-02 0.00251  3.22597E-02 0.00013  1.06495E-01 0.00053  3.00433E-01 0.00030  1.25760E+00 0.00079  5.44696E+00 0.03201 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46351E-03 0.00781  2.50162E-04 0.03015  8.02150E-04 0.01634  6.26641E-04 0.01745  1.39460E-03 0.01263  3.21938E-04 0.02747  6.80150E-05 0.05959 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.97562E-01 0.02247  1.24765E-02 3.6E-05  3.22538E-02 0.00019  1.06559E-01 0.00079  3.00317E-01 0.00043  1.25651E+00 0.00111  6.80474E+00 0.02175 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.90578E-07 0.00134  8.90177E-07 0.00134  1.00426E-06 0.01697 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.09347E-07 0.00127  8.08983E-07 0.00127  9.12614E-07 0.01693 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41869E-03 0.00927  2.40388E-04 0.03366  7.89419E-04 0.01870  6.24566E-04 0.02279  1.38432E-03 0.01456  3.12079E-04 0.02922  6.79145E-05 0.06376 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.05080E-01 0.02517  1.24765E-02 4.6E-05  3.22597E-02 0.00022  1.06517E-01 0.00097  3.00500E-01 0.00052  1.25728E+00 0.00146  6.93684E+00 0.03064 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.83507E-07 0.01707  7.83336E-07 0.01707  8.34773E-07 0.05257 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.12130E-07 0.01707  7.11974E-07 0.01707  7.58712E-07 0.05252 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06102E-03 0.04177  1.92354E-04 0.15256  7.82527E-04 0.07743  5.32790E-04 0.09208  1.19927E-03 0.06187  2.69703E-04 0.12776  8.43714E-05 0.24583 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.20811E-01 0.10532  1.24790E-02 7.6E-05  3.22267E-02 0.00067  1.06946E-01 0.00344  3.00575E-01 0.00187  1.26127E+00 0.00414  6.92865E+00 0.08228 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08801E-03 0.04108  1.99290E-04 0.15062  7.67977E-04 0.07496  5.46190E-04 0.09381  1.21974E-03 0.06076  2.70517E-04 0.12617  8.42939E-05 0.23913 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.25527E-01 0.10602  1.24791E-02 7.5E-05  3.22271E-02 0.00067  1.06885E-01 0.00338  3.00492E-01 0.00184  1.26133E+00 0.00413  6.96236E+00 0.08144 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.93614E+03 0.03852 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.82626E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.02123E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40427E-03 0.00511 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.85814E+03 0.00521 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92214E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.64539E-05 0.00926  6.64482E-05 0.00926  7.45501E-07 0.50659 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.67711E-05 0.02620  5.67553E-05 0.02622  5.44283E-07 0.62335 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.87271E-04 0.02259  1.87594E-04 0.02260  1.11906E-04 0.50063 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61922E+01 0.01069 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50413E+01 0.00026  3.99779E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 19:45:19 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01989E+00  9.97287E-01  9.99946E-01  9.99717E-01  1.00347E+00  9.89179E-01  1.00319E+00  9.99632E-01  1.00363E+00  9.99678E-01  9.98221E-01  1.00370E+00  9.99436E-01  1.00279E+00  9.91642E-01  9.95901E-01  9.92184E-01  1.00051E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.11763E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58824E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38376E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52210E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64844E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50089E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49178E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26143E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.78336E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500249 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50040E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50040E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.29901E+03 ;
RUNNING_TIME              (idx, 1)        =  4.33120E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.59777E+00  1.50750E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.29868E+02  4.24468E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.89750E-01  1.86000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.78567E-01  5.00500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.33112E+02  6.02298E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92569 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98628E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92854E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.90409E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73942E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.66546E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99827E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37805E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87169E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56765E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.17740E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17159E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.92221E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.56400E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21211E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20670E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20401E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37820E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34478E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56489E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90679E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11845E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26943E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99916E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.22117E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09954E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03839E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.26141E-02 -7.20111E+27  2.27998E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02420E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.57263E+18 0.00410  1.67853E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  8.27490E+19 0.00055  8.83284E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.91955E+18 0.00212  6.31851E-02 0.00202 ];
U238_FISS                 (idx, [1:   4]) = [  2.59196E+13 1.00000  2.80143E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  8.57646E+16 0.01837  9.15464E-04 0.01835 ];
PU240_FISS                (idx, [1:   4]) = [  3.55078E+15 0.08143  3.78666E-05 0.08130 ];
PU241_FISS                (idx, [1:   4]) = [  4.62402E+15 0.07687  4.93210E-05 0.07687 ];
TH232_CAPT                (idx, [1:   4]) = [  9.65822E+19 0.00053  6.77595E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20966E+19 0.00146  8.48691E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.09968E+18 0.00353  1.47310E-02 0.00351 ];
U238_CAPT                 (idx, [1:   4]) = [  5.14471E+14 0.21824  3.60842E-06 0.21825 ];
PU239_CAPT                (idx, [1:   4]) = [  4.07809E+16 0.02568  2.85978E-04 0.02563 ];
PU240_CAPT                (idx, [1:   4]) = [  1.10643E+16 0.04723  7.76600E-05 0.04729 ];
PU241_CAPT                (idx, [1:   4]) = [  9.32414E+14 0.16833  6.54305E-06 0.16840 ];
SM149_CAPT                (idx, [1:   4]) = [  1.76521E+17 0.01246  1.23858E-03 0.01248 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001593 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12680E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001593 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5485301 5.49074E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3605393 3.60884E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910899 9.11695E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001593 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -9.31323E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36131E+20 4.5E-06  2.36131E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36753E+19 1.1E-07  9.36753E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42452E+20 0.00028  1.27832E+20 0.00020  1.46197E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36127E+20 0.00017  2.21508E+20 0.00012  1.46197E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59597E+20 0.00034  2.59597E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45781E+22 0.00025  3.28218E+22 0.00022  1.75632E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36686E+19 0.00153 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59796E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16730E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.37367E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.37367E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.98731E-01 0.05373 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.47921E-01 0.01221 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.84807E-04 0.02436 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.38181E+03 0.02135 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08833E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.85545E-01 0.04217 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.32189E-01 0.04217 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52074E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99888E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09656E-01 0.00044  5.66596E-02 0.00044  1.96282E-04 0.00869 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09932E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09648E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09932E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00125E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92682E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92472E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47223E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47505E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90035E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90087E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.90129E-03 0.00545  2.86007E-04 0.01969  9.16277E-04 0.01039  7.02387E-04 0.01300  1.55908E-03 0.00884  3.59674E-04 0.01770  7.78723E-05 0.03511 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.03324E-01 0.01401  1.24144E-02 0.00355  3.22471E-02 0.00013  1.06640E-01 0.00053  3.00570E-01 0.00032  1.26009E+00 0.00083  5.86616E+00 0.02850 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45325E-03 0.00800  2.44637E-04 0.02829  8.17318E-04 0.01534  6.11895E-04 0.01825  1.39049E-03 0.01331  3.17898E-04 0.02524  7.10065E-05 0.05279 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09164E-01 0.02088  1.24766E-02 3.9E-05  3.22426E-02 0.00020  1.06598E-01 0.00074  3.00751E-01 0.00043  1.25916E+00 0.00099  6.94029E+00 0.02046 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.80292E-07 0.00135  8.79856E-07 0.00135  1.00345E-06 0.01513 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.00693E-07 0.00125  8.00297E-07 0.00126  9.12642E-07 0.01511 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45243E-03 0.00886  2.47239E-04 0.03290  8.14102E-04 0.01877  6.26512E-04 0.02212  1.38598E-03 0.01441  3.09816E-04 0.02909  6.87787E-05 0.06136 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.02804E-01 0.02572  1.24764E-02 5.3E-05  3.22430E-02 0.00022  1.06622E-01 0.00107  3.00751E-01 0.00058  1.26173E+00 0.00140  6.79913E+00 0.03097 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.78156E-07 0.01709  7.77760E-07 0.01710  8.98017E-07 0.05303 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.08093E-07 0.01708  7.07733E-07 0.01709  8.17386E-07 0.05300 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08794E-03 0.04266  2.29449E-04 0.13787  7.07569E-04 0.08138  5.37345E-04 0.09071  1.26279E-03 0.06603  2.99049E-04 0.12703  5.17335E-05 0.25663 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.12456E-01 0.10821  1.24768E-02 0.00015  3.22659E-02 0.00073  1.06464E-01 0.00281  2.99981E-01 0.00176  1.26130E+00 0.00363  6.88928E+00 0.09117 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08721E-03 0.04227  2.28662E-04 0.13468  7.04316E-04 0.08019  5.32435E-04 0.08942  1.26453E-03 0.06498  3.07710E-04 0.12392  4.95648E-05 0.26256 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.10349E-01 0.10593  1.24767E-02 0.00015  3.22643E-02 0.00072  1.06499E-01 0.00284  3.00008E-01 0.00174  1.26091E+00 0.00362  6.88928E+00 0.09117 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.98485E+03 0.03957 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.73910E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.94897E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53511E-03 0.00646 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.04441E+03 0.00633 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89550E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.59812E-05 0.01044  6.59650E-05 0.01043  8.27276E-07 0.52496 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.64688E-05 0.02688  5.65795E-05 0.02688  2.05928E-07 0.51978 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.78576E-04 0.02391  1.78871E-04 0.02397  9.04993E-05 0.49847 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60946E+01 0.01138 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49178E+01 0.00025  3.97804E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 20:28:47 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01687E+00  9.99930E-01  9.98244E-01  1.00011E+00  1.00041E+00  9.98963E-01  9.93998E-01  1.00387E+00  9.93907E-01  1.00032E+00  1.00067E+00  9.95383E-01  9.96415E-01  9.98636E-01  1.00092E+00  9.97075E-01  9.99002E-01  1.00527E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.13804E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58620E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38179E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52058E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65118E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49145E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48235E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24956E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.79833E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500457 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50056E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50056E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.73141E+03 ;
RUNNING_TIME              (idx, 1)        =  4.76585E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.75338E+00  1.55617E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.73118E+02  4.32497E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.09300E-01  1.95500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.29750E-01  5.11167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.76576E+02  6.04442E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92775 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98637E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93086E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.92681E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74580E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.98311E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96362E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35741E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92908E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57613E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.73873E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75295E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.24744E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03850E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44293E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70491E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34413E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37385E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33865E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55814E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95117E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27253E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26652E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05628E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41802E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07759E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04081E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.30434E-02 -1.17118E+28  2.32509E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02151E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.54186E+18 0.00399  1.64663E-02 0.00398 ];
U233_FISS                 (idx, [1:   4]) = [  8.06338E+19 0.00057  8.61108E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.92779E+18 0.00209  7.39807E-02 0.00198 ];
U238_FISS                 (idx, [1:   4]) = [  1.29071E+14 0.44499  1.38419E-06 0.44498 ];
PU239_FISS                (idx, [1:   4]) = [  2.98344E+17 0.00954  3.18592E-03 0.00951 ];
PU240_FISS                (idx, [1:   4]) = [  2.26538E+16 0.03282  2.41882E-04 0.03281 ];
PU241_FISS                (idx, [1:   4]) = [  3.27125E+16 0.02781  3.49285E-04 0.02779 ];
TH232_CAPT                (idx, [1:   4]) = [  9.43539E+19 0.00052  6.59166E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17844E+19 0.00151  8.23268E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.43831E+18 0.00316  1.70341E-02 0.00313 ];
U238_CAPT                 (idx, [1:   4]) = [  1.09574E+15 0.16102  7.65594E-06 0.16118 ];
PU239_CAPT                (idx, [1:   4]) = [  1.40821E+17 0.01442  9.83708E-04 0.01439 ];
PU240_CAPT                (idx, [1:   4]) = [  5.92357E+16 0.02159  4.13768E-04 0.02157 ];
PU241_CAPT                (idx, [1:   4]) = [  7.59576E+15 0.05540  5.30557E-05 0.05539 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78239E+17 0.01270  1.24525E-03 0.01270 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002257 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09794E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002257 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496194 5.50120E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3595686 3.59874E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910377 9.11042E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002257 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.93715E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36285E+20 4.5E-06  2.36285E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36358E+19 9.9E-08  9.36358E+19 9.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43216E+20 0.00028  1.28363E+20 0.00020  1.48522E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36851E+20 0.00017  2.21999E+20 0.00011  1.48522E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60203E+20 0.00033  2.60203E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.46617E+22 0.00024  3.28789E+22 0.00022  1.78281E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37066E+19 0.00150 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60558E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16753E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26974E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26974E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.92593E-01 0.05209 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.28322E-01 0.01210 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.09363E-04 0.02354 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.28371E+03 0.02140 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08900E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.24225E-01 0.03878 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.67270E-01 0.03878 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52345E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99972E+02 9.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.08170E-01 0.00044  5.65637E-02 0.00044  1.93607E-04 0.00931 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07852E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08118E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07852E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98826E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88025E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88376E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.54241E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53667E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.00127E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.99753E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.90415E-03 0.00571  2.78313E-04 0.02087  9.17784E-04 0.01137  6.85811E-04 0.01288  1.58001E-03 0.00861  3.59959E-04 0.01722  8.22708E-05 0.03755 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.12149E-01 0.01433  1.24464E-02 0.00251  3.22299E-02 0.00014  1.06679E-01 0.00055  3.01206E-01 0.00030  1.25940E+00 0.00095  5.75056E+00 0.02944 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45808E-03 0.00815  2.38162E-04 0.03048  8.03795E-04 0.01619  6.13138E-04 0.01916  1.41155E-03 0.01184  3.15445E-04 0.02592  7.59868E-05 0.05624 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.17253E-01 0.02244  1.24772E-02 6.8E-05  3.22328E-02 0.00019  1.06738E-01 0.00077  3.01084E-01 0.00043  1.25878E+00 0.00119  6.95811E+00 0.02032 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64438E-07 0.00134  8.64027E-07 0.00134  9.80993E-07 0.01608 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.84984E-07 0.00123  7.84611E-07 0.00124  8.90881E-07 0.01610 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39441E-03 0.00946  2.46678E-04 0.03672  7.85425E-04 0.01802  5.85448E-04 0.02276  1.39471E-03 0.01429  3.10717E-04 0.03173  7.14291E-05 0.06375 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08885E-01 0.02508  1.24781E-02 0.00010  3.22412E-02 0.00025  1.06651E-01 0.00106  3.01460E-01 0.00055  1.26127E+00 0.00145  6.70575E+00 0.03110 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63025E-07 0.01722  7.62260E-07 0.01723  8.98759E-07 0.05221 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.92892E-07 0.01721  6.92197E-07 0.01722  8.16106E-07 0.05219 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.19732E-03 0.04095  2.55831E-04 0.13505  7.88592E-04 0.08211  4.83870E-04 0.09528  1.23837E-03 0.06282  3.66741E-04 0.12376  6.39181E-05 0.28237 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.02117E-01 0.09270  1.24881E-02 0.00078  3.22234E-02 0.00098  1.06464E-01 0.00296  3.01493E-01 0.00195  1.26398E+00 0.00527  6.98836E+00 0.10130 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.19138E-03 0.04049  2.62478E-04 0.13080  7.85205E-04 0.08142  4.87180E-04 0.09277  1.22794E-03 0.06243  3.64157E-04 0.12367  6.44164E-05 0.28388 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.01730E-01 0.09517  1.24878E-02 0.00077  3.22217E-02 0.00098  1.06434E-01 0.00296  3.01496E-01 0.00194  1.26409E+00 0.00527  6.98894E+00 0.10131 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.24457E+03 0.03853 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.57828E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.78989E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.48027E-03 0.00570 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.05783E+03 0.00573 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92210E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71815E-05 0.00905  6.71740E-05 0.00905  7.01363E-07 0.51511 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.78757E-05 0.02769  5.78469E-05 0.02771  6.93328E-07 0.55726 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.01288E-04 0.02269  2.01680E-04 0.02262  1.03993E-04 0.49853 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61551E+01 0.01135 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48235E+01 0.00026  3.93445E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 21:12:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01141E+00  1.00927E+00  9.94800E-01  9.99001E-01  9.92494E-01  1.00050E+00  9.92396E-01  1.00386E+00  9.97106E-01  9.99053E-01  1.00342E+00  9.98589E-01  1.00020E+00  1.00662E+00  9.96381E-01  9.95427E-01  1.00193E+00  9.97550E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.20237E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57976E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36833E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50889E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66431E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48867E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47955E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.28681E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.86338E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500369 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50035E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50035E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.16434E+03 ;
RUNNING_TIME              (idx, 1)        =  5.20107E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.91235E+00  1.58967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.16419E+02  4.33011E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.28500E-01  1.92000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.81067E-01  5.12500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.20100E+02  6.06955E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92937 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98626E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93273E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.91453E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74028E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.64507E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92286E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33711E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95757E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57266E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.54128E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39238E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36655E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14254E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69287E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23910E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58167E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34667E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31680E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52891E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.98452E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59900E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23470E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03483E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.15819E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01702E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04460E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06399E-01 -2.34926E+28  2.44290E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01898E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.51642E+18 0.00443  1.61986E-02 0.00436 ];
U233_FISS                 (idx, [1:   4]) = [  7.93654E+19 0.00058  8.47840E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.17230E+18 0.00195  7.66189E-02 0.00185 ];
U238_FISS                 (idx, [1:   4]) = [  2.08647E+14 0.35046  2.22859E-06 0.35047 ];
PU239_FISS                (idx, [1:   4]) = [  6.13658E+17 0.00671  6.55631E-03 0.00673 ];
PU240_FISS                (idx, [1:   4]) = [  7.25670E+16 0.01833  7.74933E-04 0.01828 ];
PU241_FISS                (idx, [1:   4]) = [  1.25095E+17 0.01519  1.33661E-03 0.01520 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27465E+19 0.00053  6.44191E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15492E+19 0.00149  8.02192E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.52137E+18 0.00334  1.75141E-02 0.00337 ];
U238_CAPT                 (idx, [1:   4]) = [  2.42787E+15 0.09957  1.68619E-05 0.09956 ];
PU239_CAPT                (idx, [1:   4]) = [  2.93352E+17 0.00956  2.03757E-03 0.00956 ];
PU240_CAPT                (idx, [1:   4]) = [  1.91457E+17 0.01234  1.32973E-03 0.01233 ];
PU241_CAPT                (idx, [1:   4]) = [  2.73286E+16 0.03091  1.89889E-04 0.03093 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85199E+17 0.01177  1.28643E-03 0.01177 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001403 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.05907E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001403 1.00106E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5507830 5.51310E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3581340 3.58450E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912233 9.12993E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001403 1.00106E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.41447E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36590E+20 4.6E-06  2.36590E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35951E+19 1.0E-07  9.35951E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43907E+20 0.00030  1.28659E+20 0.00020  1.52482E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37502E+20 0.00018  2.22254E+20 0.00012  1.52482E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61151E+20 0.00034  2.61151E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49917E+22 0.00025  3.31764E+22 0.00022  1.81521E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.38444E+19 0.00160 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61347E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17101E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.06020E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.06020E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.63477E-01 0.04963 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13995E-01 0.01131 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.50432E-04 0.02200 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.84442E+03 0.02390 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08705E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.43144E-01 0.03714 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.84403E-01 0.03714 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52780E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00059E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.06035E-01 0.00045  5.64322E-02 0.00044  1.96519E-04 0.00877 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06265E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05993E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06265E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97355E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85585E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85145E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58059E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58716E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.05978E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.06659E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.95738E-03 0.00517  2.92251E-04 0.01891  9.23400E-04 0.01063  7.13349E-04 0.01251  1.57342E-03 0.00825  3.66740E-04 0.01754  8.82187E-05 0.03358 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.17753E-01 0.01417  1.24464E-02 0.00251  3.21983E-02 0.00015  1.06933E-01 0.00057  3.01412E-01 0.00030  1.25969E+00 0.00095  5.94262E+00 0.02717 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48338E-03 0.00801  2.51910E-04 0.02840  8.08496E-04 0.01672  6.21525E-04 0.01896  1.39486E-03 0.01256  3.31018E-04 0.02420  7.55699E-05 0.05281 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.13934E-01 0.02080  1.24769E-02 4.3E-05  3.21946E-02 0.00022  1.06823E-01 0.00078  3.01514E-01 0.00043  1.25973E+00 0.00127  6.84454E+00 0.02013 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.61472E-07 0.00137  8.61060E-07 0.00136  9.84496E-07 0.02161 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.80464E-07 0.00130  7.80091E-07 0.00130  8.91871E-07 0.02155 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.47179E-03 0.00885  2.51858E-04 0.03291  8.07712E-04 0.02033  6.19738E-04 0.02129  1.39227E-03 0.01406  3.21152E-04 0.02962  7.90607E-05 0.05941 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.17785E-01 0.02470  1.24785E-02 8.3E-05  3.22038E-02 0.00028  1.06685E-01 0.00094  3.01329E-01 0.00049  1.26197E+00 0.00157  6.78293E+00 0.02989 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.57672E-07 0.01704  7.57453E-07 0.01704  8.27918E-07 0.05845 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.86691E-07 0.01704  6.86493E-07 0.01705  7.50147E-07 0.05840 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.12144E-03 0.04291  2.11588E-04 0.14698  7.19241E-04 0.08151  5.97354E-04 0.09216  1.15255E-03 0.06562  3.67862E-04 0.11727  7.28429E-05 0.26406 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.26029E-01 0.09868  1.24797E-02 4.0E-05  3.21872E-02 0.00083  1.06812E-01 0.00317  3.01710E-01 0.00197  1.26585E+00 0.00547  6.81116E+00 0.09146 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10294E-03 0.04236  2.10607E-04 0.14126  7.27194E-04 0.08085  5.95205E-04 0.09152  1.13683E-03 0.06506  3.63226E-04 0.11789  6.98810E-05 0.25494 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.25905E-01 0.09810  1.24798E-02 3.3E-05  3.21908E-02 0.00083  1.06843E-01 0.00316  3.01795E-01 0.00197  1.26603E+00 0.00545  6.81116E+00 0.09146 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.14004E+03 0.03948 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.55054E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.74649E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44838E-03 0.00579 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03261E+03 0.00571 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.01134E-09 0.00091 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72425E-05 0.00746  6.72799E-05 0.00746  2.37089E-06 0.25934 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.51400E-05 0.02263  5.52439E-05 0.02263  1.60799E-06 0.35748 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.44743E-04 0.02085  2.44215E-04 0.02079  3.93881E-04 0.25441 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62442E+01 0.01109 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47955E+01 0.00028  3.90992E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 21:56:00 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01027E+00  1.00198E+00  1.00011E+00  9.99488E-01  9.96398E-01  9.99207E-01  9.91132E-01  1.00943E+00  1.00085E+00  9.95692E-01  9.99495E-01  9.94353E-01  9.97711E-01  1.00307E+00  9.97600E-01  1.00541E+00  9.94712E-01  1.00309E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.26536E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57346E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35225E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49459E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.68023E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48680E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47749E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.33517E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.93231E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500294 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50064E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50064E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.59889E+03 ;
RUNNING_TIME              (idx, 1)        =  5.63797E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.07312E+00  1.60767E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.59882E+02  4.34627E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.48683E-01  2.01833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.32533E-01  5.14167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.63789E+02  6.07271E+02 ];
CPU_USAGE                 (idx, 1)        = 9.93069 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98637E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93428E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.91193E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73844E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.41815E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91240E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33196E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96532E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57122E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64129E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56664E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45904E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16023E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76485E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39506E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93180E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33902E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31118E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52104E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99981E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00893E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22494E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02326E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34903E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00022E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04634E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61002E-01 -3.55487E+28  2.56346E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01357E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.51441E+18 0.00419  1.61740E-02 0.00409 ];
U233_FISS                 (idx, [1:   4]) = [  7.92526E+19 0.00056  8.46524E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.10632E+18 0.00191  7.59060E-02 0.00185 ];
U238_FISS                 (idx, [1:   4]) = [  2.88194E+14 0.32471  3.06764E-06 0.32468 ];
PU239_FISS                (idx, [1:   4]) = [  6.55737E+17 0.00597  7.00424E-03 0.00595 ];
PU240_FISS                (idx, [1:   4]) = [  8.69090E+16 0.01672  9.28355E-04 0.01674 ];
PU241_FISS                (idx, [1:   4]) = [  1.51988E+17 0.01299  1.62350E-03 0.01298 ];
TH232_CAPT                (idx, [1:   4]) = [  9.20613E+19 0.00057  6.39967E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15525E+19 0.00146  8.03117E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.51961E+18 0.00319  1.75159E-02 0.00319 ];
U238_CAPT                 (idx, [1:   4]) = [  2.61530E+15 0.09846  1.81932E-05 0.09852 ];
PU239_CAPT                (idx, [1:   4]) = [  3.08668E+17 0.00907  2.14586E-03 0.00908 ];
PU240_CAPT                (idx, [1:   4]) = [  2.29753E+17 0.01061  1.59725E-03 0.01063 ];
PU241_CAPT                (idx, [1:   4]) = [  3.30310E+16 0.02725  2.29619E-04 0.02725 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81865E+17 0.01200  1.26417E-03 0.01199 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002546 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08819E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002546 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494564 5.49927E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3576115 3.57901E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931867 9.32603E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002546 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.14204E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36655E+20 4.8E-06  2.36655E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35888E+19 1.1E-07  9.35888E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43868E+20 0.00028  1.28278E+20 0.00018  1.55898E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37457E+20 0.00017  2.21867E+20 0.00010  1.55898E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61586E+20 0.00034  2.61586E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.52671E+22 0.00025  3.34356E+22 0.00022  1.83152E+21 0.00186 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43962E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61853E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17245E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87896E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87896E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.83248E-01 0.04341 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.09149E-01 0.01088 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.02831E-04 0.01838 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.37815E+03 0.02518 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06745E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.23399E-01 0.03084 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.55899E-01 0.03084 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52867E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04978E-01 0.00044  5.63696E-02 0.00043  1.94663E-04 0.00886 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.04763E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.04734E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.04763E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97826E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85094E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85092E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58827E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58798E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07825E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07602E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.92130E-03 0.00534  2.80055E-04 0.01911  9.12228E-04 0.01081  7.09675E-04 0.01156  1.57982E-03 0.00812  3.56745E-04 0.01755  8.27706E-05 0.03654 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13430E-01 0.01468  1.24503E-02 0.00251  3.22036E-02 0.00015  1.06875E-01 0.00054  3.01228E-01 0.00031  1.25757E+00 0.00117  5.94072E+00 0.02918 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46951E-03 0.00772  2.54468E-04 0.02854  7.96161E-04 0.01583  6.31420E-04 0.01665  1.40843E-03 0.01212  3.09330E-04 0.02379  6.96955E-05 0.05424 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.01871E-01 0.01986  1.24810E-02 0.00019  3.21914E-02 0.00023  1.06883E-01 0.00083  3.01010E-01 0.00042  1.25897E+00 0.00151  7.20566E+00 0.01961 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.69309E-07 0.00131  8.68849E-07 0.00131  1.00649E-06 0.01930 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.86652E-07 0.00126  7.86237E-07 0.00126  9.10494E-07 0.01917 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45532E-03 0.00882  2.40553E-04 0.03275  8.02021E-04 0.01875  6.26023E-04 0.02057  1.40492E-03 0.01406  3.09596E-04 0.03087  7.21984E-05 0.06405 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06853E-01 0.02529  1.24801E-02 0.00021  3.21986E-02 0.00027  1.06998E-01 0.00101  3.01336E-01 0.00056  1.26202E+00 0.00202  6.95655E+00 0.03050 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.67826E-07 0.01711  7.67835E-07 0.01711  7.60937E-07 0.05020 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.95024E-07 0.01710  6.95032E-07 0.01711  6.88834E-07 0.05023 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.10601E-03 0.04545  2.29617E-04 0.13365  7.89733E-04 0.08328  4.72632E-04 0.09550  1.31343E-03 0.06806  2.44635E-04 0.12528  5.59606E-05 0.26086 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.50259E-01 0.11532  1.24876E-02 0.00086  3.21989E-02 0.00080  1.06673E-01 0.00306  3.00805E-01 0.00179  1.26102E+00 0.00549  7.69260E+00 0.06345 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.12038E-03 0.04542  2.30297E-04 0.13247  7.92195E-04 0.08180  4.75887E-04 0.09444  1.32343E-03 0.06858  2.42293E-04 0.12303  5.62805E-05 0.25037 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.43503E-01 0.11193  1.24876E-02 0.00086  3.21983E-02 0.00077  1.06686E-01 0.00309  3.00850E-01 0.00180  1.26100E+00 0.00549  7.69260E+00 0.06345 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.05433E+03 0.04289 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.62182E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.80199E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43509E-03 0.00659 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.98458E+03 0.00659 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.14592E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77198E-05 0.00657  6.77512E-05 0.00659  1.74595E-06 0.30464 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.87344E-05 0.01887  5.87698E-05 0.01887  1.11471E-06 0.37713 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.90766E-04 0.01807  2.90804E-04 0.01806  2.87187E-04 0.29989 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63819E+01 0.01141 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47749E+01 0.00027  3.90417E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 12:32:12 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 22:22:13 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651246332472 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02129E+00  9.93925E-01  9.99419E-01  1.00085E+00  9.89254E-01  9.88803E-01  9.93977E-01  1.00223E+00  1.00746E+00  9.96094E-01  9.99909E-01  1.00164E+00  9.97505E-01  9.96009E-01  1.00267E+00  1.00175E+00  1.00274E+00  1.00448E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.33707E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56629E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.33838E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.48271E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69429E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49191E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48251E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.38831E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  3.01054E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500735 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50063E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50063E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.85874E+03 ;
RUNNING_TIME              (idx, 1)        =  5.90010E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.17868E+00  1.17868E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.23430E+00  1.61183E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.85866E+02  2.59839E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.68200E-01  1.95167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.82550E-01  4.99667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.90002E+02  5.90002E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92990 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97940E+00 3.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93374E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22876.42;
MEMSIZE                   (idx, 1)        = 22582.79;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 431.08;
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

TOT_ACTIVITY              (idx, 1)        =  6.90747E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73761E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.64373E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90555E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32788E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96766E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57075E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65608E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61374E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47158E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16146E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78567E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44061E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03111E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33670E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30949E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51866E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00268E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13450E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21892E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01947E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.39098E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98664E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.04771E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16244E-01 -4.77460E+28  2.68543E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01091E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.51519E+18 0.00436  1.61900E-02 0.00431 ];
U233_FISS                 (idx, [1:   4]) = [  7.91984E+19 0.00057  8.46274E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.09915E+18 0.00190  7.58577E-02 0.00183 ];
U238_FISS                 (idx, [1:   4]) = [  2.88865E+14 0.29775  3.07962E-06 0.29773 ];
PU239_FISS                (idx, [1:   4]) = [  6.63271E+17 0.00606  7.08803E-03 0.00607 ];
PU240_FISS                (idx, [1:   4]) = [  8.91455E+16 0.01701  9.52727E-04 0.01702 ];
PU241_FISS                (idx, [1:   4]) = [  1.59257E+17 0.01326  1.70165E-03 0.01325 ];
TH232_CAPT                (idx, [1:   4]) = [  9.17164E+19 0.00058  6.37027E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15495E+19 0.00154  8.02191E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49761E+18 0.00310  1.73481E-02 0.00309 ];
U238_CAPT                 (idx, [1:   4]) = [  2.92818E+15 0.09133  2.03493E-05 0.09140 ];
PU239_CAPT                (idx, [1:   4]) = [  3.13193E+17 0.00975  2.17550E-03 0.00975 ];
PU240_CAPT                (idx, [1:   4]) = [  2.30080E+17 0.01054  1.59801E-03 0.01053 ];
PU241_CAPT                (idx, [1:   4]) = [  3.41580E+16 0.02765  2.37316E-04 0.02767 ];
SM149_CAPT                (idx, [1:   4]) = [  1.86267E+17 0.01138  1.29374E-03 0.01137 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002515 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08066E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002515 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5492056 5.49678E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3570120 3.57295E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 940339 9.41077E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002515 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.44007E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36665E+20 4.9E-06  2.36665E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35881E+19 9.9E-08  9.35881E+19 9.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43953E+20 0.00028  1.27978E+20 0.00018  1.59747E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37541E+20 0.00017  2.21566E+20 0.00010  1.59747E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61928E+20 0.00035  2.61928E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.55645E+22 0.00024  3.37101E+22 0.00022  1.85440E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46503E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62191E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17528E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.72235E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.72235E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.10449E-01 0.03639 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.04459E-01 0.00986 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.65198E-04 0.01714 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.93091E+03 0.02515 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.05899E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 9.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.22416E-01 0.02306 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.45007E-01 0.02306 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52880E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00074E+02 9.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03415E-01 0.00043  5.62751E-02 0.00043  1.94000E-04 0.00933 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.03626E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03595E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.03626E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97505E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85044E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85208E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.58910E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.58617E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.07721E-01 0.00147 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.07713E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.95692E-03 0.00573  2.72278E-04 0.02008  9.09482E-04 0.01074  7.21421E-04 0.01196  1.59522E-03 0.00859  3.78162E-04 0.01721  8.03613E-05 0.03644 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10863E-01 0.01367  1.24795E-02 8.4E-05  3.21995E-02 0.00016  1.06876E-01 0.00053  3.01323E-01 0.00031  1.25860E+00 0.00100  6.02292E+00 0.02787 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46090E-03 0.00799  2.36147E-04 0.03056  7.87979E-04 0.01608  6.27208E-04 0.01800  1.41163E-03 0.01226  3.29123E-04 0.02442  6.88129E-05 0.05247 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.08294E-01 0.01969  1.24794E-02 0.00011  3.22063E-02 0.00023  1.06896E-01 0.00077  3.01252E-01 0.00045  1.25983E+00 0.00123  6.97225E+00 0.02023 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.78121E-07 0.00148  8.77660E-07 0.00149  1.01144E-06 0.01976 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.93234E-07 0.00138  7.92818E-07 0.00139  9.13485E-07 0.01969 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43744E-03 0.00933  2.30120E-04 0.03559  7.93387E-04 0.01844  6.40741E-04 0.01973  1.37419E-03 0.01416  3.26088E-04 0.03004  7.29181E-05 0.06519 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.12258E-01 0.02494  1.24851E-02 0.00027  3.22037E-02 0.00028  1.06941E-01 0.00098  3.01406E-01 0.00055  1.26043E+00 0.00190  6.98238E+00 0.02955 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.58805E-07 0.01719  7.58413E-07 0.01719  8.44371E-07 0.05706 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.85678E-07 0.01717  6.85323E-07 0.01717  7.62971E-07 0.05710 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.99538E-03 0.04191  2.02941E-04 0.15760  6.44419E-04 0.08476  6.34522E-04 0.08604  1.19871E-03 0.06393  2.51739E-04 0.12772  6.30507E-05 0.24182 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.97363E-01 0.10626  1.24889E-02 0.00086  3.22423E-02 0.00110  1.06704E-01 0.00297  3.00726E-01 0.00184  1.26265E+00 0.00408  6.57421E+00 0.08760 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98000E-03 0.04140  2.07485E-04 0.15633  6.33964E-04 0.08351  6.29796E-04 0.08365  1.19785E-03 0.06298  2.46797E-04 0.12716  6.41055E-05 0.24264 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.94966E-01 0.10689  1.24889E-02 0.00086  3.22441E-02 0.00110  1.06714E-01 0.00298  3.00714E-01 0.00183  1.26299E+00 0.00411  6.56451E+00 0.08786 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96666E+03 0.03823 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.65679E-07 0.00080 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.82008E-07 0.00067 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44922E-03 0.00598 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.98532E+03 0.00601 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.30005E-09 0.00099 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.91538E-05 0.00528  6.91718E-05 0.00531  1.72882E-06 0.30196 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.13211E-05 0.01703  6.14037E-05 0.01706  9.66845E-07 0.43686 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.55412E-04 0.01666  3.55619E-04 0.01658  3.34505E-04 0.30914 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58208E+01 0.01126 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48251E+01 0.00028  3.90268E+01 0.00037 ];

