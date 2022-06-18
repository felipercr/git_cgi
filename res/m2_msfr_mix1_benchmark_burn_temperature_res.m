
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 15:56:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02197E+00  1.00466E+00  9.94409E-01  9.99190E-01  9.97769E-01  9.94266E-01  1.00792E+00  9.96069E-01  1.00229E+00  1.00100E+00  9.98161E-01  9.87574E-01  9.99518E-01  9.89186E-01  1.00718E+00  1.00635E+00  9.88701E-01  1.00379E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.67176E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63282E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40248E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54073E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64224E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45674E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44768E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.11386E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.14240E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333863 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33403E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33403E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.87577E+02 ;
RUNNING_TIME              (idx, 1)        =  3.09174E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.71683E-01  1.71683E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.96991E+01  2.96991E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.35090E+00  1.30227E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.96548E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.30149 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98540E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.25074E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

NORM_COEF                 (idx, [1:   4]) = [  7.14963E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.84302E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.55598E+18 0.00416  1.65943E-02 0.00413 ];
U233_FISS                 (idx, [1:   4]) = [  9.22107E+19 0.00050  9.83406E-01 7.0E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.34290E+19 0.00050  7.58324E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34394E+19 0.00134  1.09082E-01 0.00128 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002091 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15414E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002091 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5164664 5.16973E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3930880 3.93449E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906547 9.07318E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002091 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.68221E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36450E+20 3.7E-06  2.36450E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37767E+19 1.2E-07  9.37767E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23269E+20 0.00030  1.10022E+20 0.00022  1.32473E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17046E+20 0.00017  2.03799E+20 0.00012  1.32473E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38321E+20 0.00036  2.38321E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.14394E+22 0.00025  2.98236E+22 0.00023  1.61583E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.16240E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38670E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06114E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.29401E-01 0.09207 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.88597E-01 0.00867 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.09377E-04 0.01556 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.17581E+04 0.02385 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09276E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.03504E-01 0.07547 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.66920E-01 0.07547 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52142E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.91905E-01 0.00040  1.09873E-01 0.00039  3.55548E-04 0.00862 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.91856E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.92189E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.91856E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09079E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01823E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02039E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34347E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34040E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49061E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48474E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.24221E-03 0.00576  2.63853E-04 0.01988  7.86522E-04 0.01250  5.83979E-04 0.01329  1.27992E-03 0.00866  2.74863E-04 0.01823  5.30743E-05 0.04148 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.57267E-01 0.01517  1.24756E-02 3.0E-05  3.23352E-02 9.2E-05  1.06178E-01 0.00060  2.97364E-01 0.00026  1.23520E+00 0.00022  5.49146E+00 0.03682 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20281E-03 0.00857  2.68685E-04 0.03021  7.84937E-04 0.01801  5.78029E-04 0.01915  1.25701E-03 0.01233  2.62379E-04 0.02905  5.17722E-05 0.06296 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.53853E-01 0.02275  1.24758E-02 3.9E-05  3.23360E-02 0.00013  1.06227E-01 0.00093  2.97304E-01 0.00038  1.23565E+00 0.00029  6.61709E+00 0.02869 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.20453E-07 0.00121  8.20198E-07 0.00121  8.99886E-07 0.01555 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.13771E-07 0.00113  8.13519E-07 0.00114  8.92370E-07 0.01547 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21976E-03 0.00876  2.65915E-04 0.03291  7.95729E-04 0.01806  5.69718E-04 0.02042  1.27899E-03 0.01442  2.56814E-04 0.03092  5.25916E-05 0.07208 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47622E-01 0.02539  1.24747E-02 5.8E-05  3.23371E-02 0.00015  1.06106E-01 0.00102  2.97312E-01 0.00041  1.23626E+00 0.00036  6.33978E+00 0.03986 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.34250E-07 0.01971  7.33920E-07 0.01972  8.04514E-07 0.05347 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.28375E-07 0.01971  7.28049E-07 0.01972  7.98062E-07 0.05343 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.10115E-03 0.04110  3.41088E-04 0.11145  6.66260E-04 0.09238  6.21156E-04 0.07764  1.16302E-03 0.06229  2.62181E-04 0.11903  4.74425E-05 0.24507 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.41510E-01 0.08266  1.24733E-02 0.00016  3.23137E-02 0.00042  1.05652E-01 0.00240  2.97453E-01 0.00162  1.23794E+00 0.00088  5.88996E+00 0.11883 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08268E-03 0.04049  3.36989E-04 0.11150  6.67456E-04 0.09162  6.06720E-04 0.07676  1.16085E-03 0.06132  2.63745E-04 0.11496  4.69261E-05 0.24222 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.46140E-01 0.07815  1.24733E-02 0.00016  3.23141E-02 0.00042  1.05664E-01 0.00237  2.97444E-01 0.00159  1.23794E+00 0.00088  5.88996E+00 0.11883 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.25756E+03 0.03667 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.15240E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08599E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.31784E-03 0.00507 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.07145E+03 0.00528 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.05857E-09 0.00119 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.23020E-05 0.00505  7.23104E-05 0.00505  1.74956E-06 0.36098 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.99875E-05 0.01765  8.00310E-05 0.01773  1.93408E-06 0.50770 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.04259E-04 0.01517  4.04778E-04 0.01513  2.68989E-04 0.35086 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69304E+01 0.01226 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44768E+01 0.00026  4.01458E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 16:42:55 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00842E+00  9.93762E-01  9.97579E-01  9.96334E-01  1.00163E+00  9.99646E-01  1.00882E+00  9.99543E-01  1.00105E+00  9.88472E-01  9.94943E-01  1.00290E+00  9.99342E-01  9.95271E-01  1.00783E+00  1.00525E+00  9.93488E-01  1.00572E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.66514E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63349E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40169E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53979E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64272E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46290E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45385E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.12845E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.14013E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333347 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33364E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33364E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.29091E+02 ;
RUNNING_TIME              (idx, 1)        =  7.71017E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.89883E-01  1.18217E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.57402E+01  4.60411E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.92333E-02  1.92333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.48753E+00  2.13658E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.50057E+01  4.62881E+02 ];
CPU_USAGE                 (idx, 1)        = 9.45622 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98933E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.41269E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  5.79263E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60555E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.66098E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.49652E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.05411E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26523E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46919E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.48818E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.66218E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.42503E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.83986E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06315E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17819E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44272E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35655E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17539E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55429E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59940E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38166E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.40723E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51679E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51643E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.33862E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.22009E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08074E-04 -2.38625E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.96000E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.56465E+18 0.00397  1.66940E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  9.21291E+19 0.00046  9.83007E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.84408E+14 0.24366  4.10021E-06 0.24366 ];
TH232_CAPT                (idx, [1:   4]) = [  9.45493E+19 0.00053  7.53836E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34574E+19 0.00135  1.07297E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  1.92851E+14 0.34941  1.53695E-06 0.34941 ];
SM149_CAPT                (idx, [1:   4]) = [  1.25251E+16 0.04149  9.97730E-05 0.04139 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000916 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15705E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000916 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5205734 5.21150E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3890210 3.89425E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904972 9.05822E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000916 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.62633E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36443E+20 3.7E-06  2.36443E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.2E-07  9.37764E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25431E+20 0.00029  1.12046E+20 0.00023  1.33852E+19 0.00171 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19208E+20 0.00017  2.05823E+20 0.00013  1.33852E+19 0.00171 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40670E+20 0.00036  2.40670E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.17838E+22 0.00025  3.01541E+22 0.00022  1.62971E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.18013E+19 0.00153 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41009E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07308E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.64259E-01 0.07895 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.81460E-01 0.00948 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.66035E-04 0.01725 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.16697E+04 0.02175 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09424E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 9.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.82125E-01 0.06437 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.38328E-01 0.06437 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52135E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.81737E-01 0.00040  1.08753E-01 0.00039  3.46002E-04 0.00865 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.82203E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.82478E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.82203E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08001E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02070E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02139E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34012E-02 0.00126 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33903E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49395E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49043E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.25736E-03 0.00556  2.62963E-04 0.01923  8.08493E-04 0.01132  5.96146E-04 0.01237  1.26891E-03 0.00902  2.71934E-04 0.01835  4.89096E-05 0.04604 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43416E-01 0.01471  1.24754E-02 2.8E-05  3.23338E-02 8.6E-05  1.06088E-01 0.00056  2.97379E-01 0.00026  1.23531E+00 0.00022  5.18158E+00 0.04115 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.16709E-03 0.00845  2.55798E-04 0.02819  7.89257E-04 0.01700  5.83858E-04 0.01941  1.21909E-03 0.01345  2.71727E-04 0.02680  4.73673E-05 0.06134 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45955E-01 0.02099  1.24757E-02 3.6E-05  3.23332E-02 0.00012  1.06127E-01 0.00082  2.97365E-01 0.00039  1.23572E+00 0.00030  6.45480E+00 0.03097 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.27762E-07 0.00110  8.27416E-07 0.00111  9.44701E-07 0.02343 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.12610E-07 0.00105  8.12271E-07 0.00105  9.27003E-07 0.02327 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.17331E-03 0.00870  2.55421E-04 0.03232  7.74247E-04 0.01838  5.77401E-04 0.02030  1.25156E-03 0.01436  2.63410E-04 0.02960  5.12686E-05 0.06949 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.56017E-01 0.02595  1.24753E-02 5.0E-05  3.23286E-02 0.00013  1.06149E-01 0.00103  2.97118E-01 0.00043  1.23559E+00 0.00039  6.53573E+00 0.04019 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.34880E-07 0.01959  7.34475E-07 0.01959  8.15332E-07 0.05786 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21558E-07 0.01959  7.21161E-07 0.01959  8.00360E-07 0.05781 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.79656E-03 0.04403  2.45706E-04 0.12290  6.94857E-04 0.07715  4.53610E-04 0.09370  1.17780E-03 0.06396  2.02587E-04 0.13116  2.20037E-05 0.33265 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.06205E-01 0.09851  1.24733E-02 0.00017  3.23092E-02 0.00035  1.06002E-01 0.00339  2.97933E-01 0.00175  1.23598E+00 0.00109  6.26138E+00 0.15198 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81831E-03 0.04350  2.47889E-04 0.12180  7.12392E-04 0.07464  4.46252E-04 0.09394  1.18224E-03 0.06339  2.04614E-04 0.12668  2.49161E-05 0.33538 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.05041E-01 0.09138  1.24732E-02 0.00017  3.23093E-02 0.00035  1.06031E-01 0.00340  2.97986E-01 0.00174  1.23587E+00 0.00110  6.26138E+00 0.15198 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81983E+03 0.03970 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.23015E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.07946E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.16334E-03 0.00582 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84402E+03 0.00585 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.00535E-09 0.00116 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.16633E-05 0.00509  7.16516E-05 0.00511  2.03414E-06 0.36417 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.69056E-05 0.01692  7.67761E-05 0.01700  2.90779E-06 0.40961 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.57762E-04 0.01660  3.58130E-04 0.01658  2.62440E-04 0.35231 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72081E+01 0.01152 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45385E+01 0.00024  4.01918E+01 0.00032 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 17:31:57 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01704E+00  9.96062E-01  9.97654E-01  9.99373E-01  9.94107E-01  9.98384E-01  1.00026E+00  9.89483E-01  9.97242E-01  1.00145E+00  9.91653E-01  1.00077E+00  1.01128E+00  1.00437E+00  9.92265E-01  1.00358E+00  1.00183E+00  1.00320E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.64714E-02 0.00037  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63529E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40567E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54309E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63642E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47392E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46487E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.13682E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.13138E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333944 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33383E+04 0.00054 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33383E+04 0.00054 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.19789E+03 ;
RUNNING_TIME              (idx, 1)        =  1.26125E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.18367E-01  1.28483E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.24608E+02  4.88677E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.81333E-02  1.89000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.74522E+00  2.25763E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.23915E+02  6.78177E+02 ];
CPU_USAGE                 (idx, 1)        = 9.49759 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98973E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.45585E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.41958E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65763E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.23945E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88713E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31892E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.50128E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49450E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.81588E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.41356E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.91637E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.10574E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89930E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60298E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49767E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34960E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33618E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54687E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.69939E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.18125E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.03393E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51947E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.33180E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.57580E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.31786E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.73286E-04 -1.26580E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.10803E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.56351E+18 0.00384  1.66526E-02 0.00375 ];
U233_FISS                 (idx, [1:   4]) = [  9.21954E+19 0.00051  9.82020E-01 6.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.06350E+16 0.04854  1.13245E-04 0.04852 ];
TH232_CAPT                (idx, [1:   4]) = [  9.59082E+19 0.00050  7.48043E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34463E+19 0.00131  1.04877E-01 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  4.36599E+15 0.07133  3.40445E-05 0.07120 ];
SM149_CAPT                (idx, [1:   4]) = [  5.91149E+16 0.01985  4.61053E-04 0.01983 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001495 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14389E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001495 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5250747 5.25616E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3845095 3.84882E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905653 9.06465E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001495 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.07919E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36436E+20 3.6E-06  2.36436E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37759E+19 1.2E-07  9.37759E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28180E+20 0.00028  1.14543E+20 0.00020  1.36375E+19 0.00171 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.21956E+20 0.00016  2.08319E+20 0.00011  1.36375E+19 0.00171 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43929E+20 0.00034  2.43929E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.22317E+22 0.00023  3.05704E+22 0.00021  1.66129E+21 0.00169 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.21118E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44068E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09027E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51428E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51428E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.60762E-01 0.05709 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.96930E-01 0.01249 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.32686E-04 0.02262 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.76572E+03 0.02661 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09358E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 6.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.96617E-01 0.04213 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.33408E-01 0.04213 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52128E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.70271E-01 0.00040  1.07470E-01 0.00040  3.51666E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.69862E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.69315E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.69862E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06660E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02612E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02346E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33287E-02 0.00124 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33627E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49424E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50194E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33380E-03 0.00589  2.64228E-04 0.01961  8.15182E-04 0.01102  6.06460E-04 0.01364  1.32303E-03 0.00893  2.74966E-04 0.01943  4.99385E-05 0.04388 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42289E-01 0.01469  1.24757E-02 2.8E-05  3.23359E-02 9.5E-05  1.06180E-01 0.00060  2.97345E-01 0.00025  1.23535E+00 0.00023  5.18170E+00 0.03928 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24542E-03 0.00840  2.57812E-04 0.02962  8.04817E-04 0.01641  5.85440E-04 0.01969  1.28556E-03 0.01290  2.66101E-04 0.02756  4.56910E-05 0.06416 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38589E-01 0.02174  1.24759E-02 3.4E-05  3.23331E-02 0.00013  1.06103E-01 0.00082  2.97389E-01 0.00039  1.23541E+00 0.00031  6.38276E+00 0.03100 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.39416E-07 0.00118  8.39135E-07 0.00118  9.25986E-07 0.01626 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14414E-07 0.00108  8.14141E-07 0.00108  8.98358E-07 0.01625 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27054E-03 0.00875  2.68614E-04 0.03002  8.06085E-04 0.01757  5.74518E-04 0.02205  1.28887E-03 0.01420  2.80902E-04 0.03033  5.15531E-05 0.06788 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44473E-01 0.02253  1.24766E-02 3.8E-05  3.23438E-02 0.00018  1.06102E-01 0.00101  2.97477E-01 0.00042  1.23569E+00 0.00038  5.99163E+00 0.04268 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.42256E-07 0.01958  7.42273E-07 0.01958  7.09432E-07 0.05317 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20312E-07 0.01957  7.20328E-07 0.01957  6.88889E-07 0.05316 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.05739E-03 0.04532  2.76874E-04 0.11878  7.71360E-04 0.07685  5.05915E-04 0.09092  1.20058E-03 0.06683  2.80125E-04 0.12583  2.25287E-05 0.32476 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.18117E-01 0.08383  1.24760E-02 0.00013  3.23520E-02 0.00061  1.06054E-01 0.00319  2.97639E-01 0.00172  1.23339E+00 0.00117  8.08993E+00 0.11418 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08633E-03 0.04461  2.81958E-04 0.11704  7.71418E-04 0.07498  5.11769E-04 0.08863  1.21508E-03 0.06542  2.84250E-04 0.12416  2.18505E-05 0.30749 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.13946E-01 0.07716  1.24759E-02 0.00013  3.23509E-02 0.00061  1.06078E-01 0.00324  2.97645E-01 0.00171  1.23355E+00 0.00115  8.08993E+00 0.11418 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.13777E+03 0.04126 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.33010E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08204E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29264E-03 0.00611 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95277E+03 0.00606 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.84155E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.95828E-05 0.00701  6.92847E-05 0.00773  1.45662E-06 0.41165 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.58046E-05 0.02326  6.52575E-05 0.02288  1.27637E-06 0.57861 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.31075E-04 0.02208  2.31246E-04 0.02222  1.84486E-04 0.40533 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70772E+01 0.01110 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46487E+01 0.00025  4.03314E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 18:21:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00776E+00  9.98252E-01  9.92843E-01  1.00203E+00  1.00094E+00  1.00372E+00  9.95798E-01  1.00004E+00  9.98723E-01  1.00079E+00  9.95602E-01  1.00423E+00  1.00300E+00  9.99453E-01  9.98404E-01  9.97733E-01  1.00156E+00  9.99120E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.63435E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63656E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40540E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54238E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.63440E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48271E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47367E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.15536E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.12737E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333611 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33374E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33374E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.67278E+03 ;
RUNNING_TIME              (idx, 1)        =  1.75778E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.46333E-01  1.27967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.74102E+02  4.94942E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.61000E-02  1.79667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.01958E+00  2.27433E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.73548E+02  7.12697E+02 ];
CPU_USAGE                 (idx, 1)        = 9.51644 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98987E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.47557E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.53648E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67396E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.15855E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92246E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34268E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58246E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50808E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.86893E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.57539E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.06854E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.29097E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79930E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74627E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48905E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36807E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35042E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56601E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39984E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42652E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10695E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55034E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.16724E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.73418E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.36802E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.11978E-03 -2.47245E+26  2.21044E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.24593E-01 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.59153E+18 0.00412  1.69776E-02 0.00405 ];
U233_FISS                 (idx, [1:   4]) = [  9.18908E+19 0.00049  9.80274E-01 7.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.28019E+16 0.02389  4.56674E-04 0.02390 ];
TH232_CAPT                (idx, [1:   4]) = [  9.67958E+19 0.00052  7.45159E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34422E+19 0.00131  1.03482E-01 0.00126 ];
U235_CAPT                 (idx, [1:   4]) = [  1.46858E+16 0.04372  1.13089E-04 0.04372 ];
SM149_CAPT                (idx, [1:   4]) = [  9.57918E+16 0.01665  7.37477E-04 0.01667 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001221 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16145E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001221 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5283433 5.28909E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3812899 3.81679E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904889 9.05733E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001221 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.32248E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36430E+20 3.7E-06  2.36430E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 1.2E-07  9.37751E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29821E+20 0.00031  1.16109E+20 0.00023  1.37115E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23596E+20 0.00018  2.09884E+20 0.00012  1.37115E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.45601E+20 0.00035  2.45601E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24853E+22 0.00025  3.08175E+22 0.00023  1.66779E+21 0.00196 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.22454E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.45841E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09993E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51430E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51430E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.95011E-01 0.05510 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.14386E-01 0.01343 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.77936E-04 0.02383 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.82805E+03 0.02602 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09430E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.05490E-01 0.04090 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.41530E-01 0.04090 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52124E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.62230E-01 0.00042  1.06583E-01 0.00041  3.40524E-04 0.00954 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.62839E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.62694E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.62839E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05875E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02661E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02681E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33228E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33182E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51494E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51173E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.36997E-03 0.00539  2.75853E-04 0.01929  8.15507E-04 0.01125  6.19340E-04 0.01294  1.32849E-03 0.00889  2.75968E-04 0.01974  5.48111E-05 0.04659 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50590E-01 0.01577  1.24759E-02 2.6E-05  3.23350E-02 9.5E-05  1.06128E-01 0.00060  2.97523E-01 0.00027  1.23542E+00 0.00025  5.15010E+00 0.03994 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23280E-03 0.00822  2.72233E-04 0.02864  7.80636E-04 0.01569  5.90244E-04 0.01939  1.27965E-03 0.01290  2.57152E-04 0.03031  5.28896E-05 0.06622 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.49322E-01 0.02300  1.24760E-02 3.6E-05  3.23391E-02 0.00014  1.06140E-01 0.00084  2.97669E-01 0.00041  1.23498E+00 0.00033  6.45574E+00 0.02936 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.48658E-07 0.00121  8.48342E-07 0.00121  9.46935E-07 0.01587 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16560E-07 0.00112  8.16255E-07 0.00113  9.11294E-07 0.01590 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19355E-03 0.00965  2.60647E-04 0.03257  7.85805E-04 0.01741  5.80793E-04 0.02262  1.26354E-03 0.01591  2.51755E-04 0.03291  5.10077E-05 0.07870 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47949E-01 0.02682  1.24750E-02 5.4E-05  3.23340E-02 0.00017  1.05999E-01 0.00098  2.97498E-01 0.00044  1.23532E+00 0.00045  6.70986E+00 0.04114 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.52203E-07 0.01965  7.51824E-07 0.01966  8.46216E-07 0.05753 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23738E-07 0.01965  7.23374E-07 0.01966  8.13775E-07 0.05747 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.84674E-03 0.04210  2.07733E-04 0.12857  6.46272E-04 0.07576  4.90658E-04 0.09426  1.21493E-03 0.06042  2.53570E-04 0.14284  3.35837E-05 0.27419 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.28702E-01 0.07516  1.24766E-02 0.00013  3.23395E-02 0.00056  1.05720E-01 0.00279  2.97660E-01 0.00164  1.23422E+00 0.00126  6.44147E+00 0.11695 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85209E-03 0.04201  2.10505E-04 0.12826  6.68717E-04 0.07439  4.85486E-04 0.09236  1.20774E-03 0.05936  2.45081E-04 0.13972  3.45593E-05 0.28326 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.23912E-01 0.07330  1.24766E-02 0.00013  3.23382E-02 0.00055  1.05693E-01 0.00270  2.97704E-01 0.00163  1.23416E+00 0.00126  6.44147E+00 0.11695 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.80021E+03 0.03797 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.42128E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10283E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17936E-03 0.00548 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77615E+03 0.00555 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78408E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68916E-05 0.00833  6.69161E-05 0.00831  1.99070E-06 0.31900 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.41627E-05 0.02410  6.38804E-05 0.02432  3.09438E-06 0.37927 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.74622E-04 0.02313  1.74212E-04 0.02332  2.98109E-04 0.31262 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71625E+01 0.01196 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47367E+01 0.00028  4.04053E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 19:13:03 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00602E+00  9.99047E-01  1.00116E+00  9.99875E-01  9.99409E-01  9.97753E-01  1.00413E+00  9.98669E-01  9.97537E-01  9.91989E-01  9.99527E-01  1.00080E+00  1.00185E+00  1.00030E+00  1.00214E+00  9.96905E-01  1.00794E+00  9.94945E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.62861E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63714E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40441E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54131E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62807E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49720E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48820E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18568E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.12862E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333249 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33383E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33383E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.15816E+03 ;
RUNNING_TIME              (idx, 1)        =  2.27236E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.77717E-01  1.31383E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.25396E+02  5.12939E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.40833E-02  1.79833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.11149E+01  3.09527E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.24182E+02  7.20618E+02 ];
CPU_USAGE                 (idx, 1)        = 9.49744 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98932E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.45439E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.56540E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67014E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.51263E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94133E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35421E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.59231E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50292E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00505E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.69791E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.33559E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.37702E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.70919E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.86009E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93949E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33752E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32706E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53373E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35676E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81741E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10617E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52872E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51169E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.77903E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.45260E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.26214E-03 -4.99474E+26  2.21297E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.46410E-01 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.60311E+18 0.00378  1.71013E-02 0.00374 ];
U233_FISS                 (idx, [1:   4]) = [  9.15716E+19 0.00056  9.76855E-01 7.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.57160E+17 0.01157  1.67642E-03 0.01153 ];
TH232_CAPT                (idx, [1:   4]) = [  9.82488E+19 0.00050  7.41119E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33666E+19 0.00133  1.00830E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  5.83863E+16 0.01911  4.40456E-04 0.01911 ];
SM149_CAPT                (idx, [1:   4]) = [  1.34344E+17 0.01410  1.01342E-03 0.01409 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001492 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15138E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001492 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5331004 5.33651E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3769816 3.77351E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 900672 9.01494E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001492 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.82311E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36418E+20 3.6E-06  2.36418E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37730E+19 1.1E-07  9.37730E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32494E+20 0.00030  1.18619E+20 0.00021  1.38746E+19 0.00193 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26267E+20 0.00017  2.12392E+20 0.00012  1.38746E+19 0.00193 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48420E+20 0.00036  2.48420E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29334E+22 0.00026  3.12508E+22 0.00023  1.68259E+21 0.00202 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.23959E+19 0.00162 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.48663E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11615E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51152E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51152E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.42835E-01 0.05542 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.20592E-01 0.01334 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.56871E-04 0.02496 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.98804E+03 0.02473 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09854E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.83513E-01 0.04213 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.21986E-01 0.04213 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52117E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.51422E-01 0.00047  1.05368E-01 0.00046  3.36102E-04 0.00922 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.51867E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.51723E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.51867E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04620E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02643E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02524E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33245E-02 0.00124 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33389E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52661E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53399E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39979E-03 0.00590  2.84383E-04 0.01907  8.29695E-04 0.01076  6.09419E-04 0.01339  1.33693E-03 0.00889  2.87117E-04 0.01953  5.22406E-05 0.04499 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.49478E-01 0.01570  1.24752E-02 3.0E-05  3.23355E-02 0.00010  1.06130E-01 0.00058  2.97694E-01 0.00028  1.23631E+00 0.00032  5.10621E+00 0.04062 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21256E-03 0.00833  2.62312E-04 0.02643  7.77278E-04 0.01586  5.79921E-04 0.01929  1.27951E-03 0.01309  2.63568E-04 0.02757  4.99757E-05 0.06544 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.47797E-01 0.02181  1.24759E-02 3.4E-05  3.23373E-02 0.00015  1.06070E-01 0.00074  2.97638E-01 0.00037  1.23607E+00 0.00035  6.30178E+00 0.03087 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.58231E-07 0.00120  8.57885E-07 0.00120  9.62707E-07 0.01670 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16481E-07 0.00109  8.16152E-07 0.00109  9.15923E-07 0.01669 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18425E-03 0.00934  2.77748E-04 0.03301  7.84406E-04 0.01811  5.55847E-04 0.02191  1.25687E-03 0.01487  2.62944E-04 0.03158  4.64359E-05 0.07157 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.40813E-01 0.02423  1.24744E-02 6.4E-05  3.23393E-02 0.00020  1.06094E-01 0.00103  2.97554E-01 0.00045  1.23711E+00 0.00068  6.28094E+00 0.04370 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.56305E-07 0.01957  7.55833E-07 0.01957  9.03744E-07 0.06188 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.19639E-07 0.01956  7.19189E-07 0.01956  8.60162E-07 0.06195 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87044E-03 0.04401  2.33850E-04 0.13467  6.72025E-04 0.07890  5.47620E-04 0.09184  1.13033E-03 0.06703  2.24541E-04 0.13497  6.20724E-05 0.28856 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.74467E-01 0.11510  1.24728E-02 0.00019  3.23492E-02 0.00068  1.06226E-01 0.00357  2.97776E-01 0.00177  1.23361E+00 0.00124  7.20878E+00 0.10165 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85204E-03 0.04277  2.32510E-04 0.13390  6.61273E-04 0.07705  5.55983E-04 0.08948  1.11014E-03 0.06547  2.27984E-04 0.12933  6.41491E-05 0.28011 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.80972E-01 0.11158  1.24728E-02 0.00019  3.23483E-02 0.00067  1.06212E-01 0.00354  2.97780E-01 0.00175  1.23352E+00 0.00124  7.20878E+00 0.10165 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.82161E+03 0.04004 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.52734E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11253E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19474E-03 0.00524 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.74752E+03 0.00536 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76929E-09 0.00069 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72115E-05 0.00930  6.72228E-05 0.00931  7.32732E-07 0.53626 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.93081E-05 0.02900  5.93657E-05 0.02911  8.09874E-07 0.57529 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.55644E-04 0.02350  1.55774E-04 0.02354  1.22140E-04 0.49853 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72009E+01 0.01173 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48820E+01 0.00028  4.04975E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 20:04:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01563E+00  1.00438E+00  1.00137E+00  1.00244E+00  9.92872E-01  9.97698E-01  9.97428E-01  1.00269E+00  9.95111E-01  9.97776E-01  9.95728E-01  9.99422E-01  1.00127E+00  9.98423E-01  1.00180E+00  9.99040E-01  9.95047E-01  1.00188E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.62506E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63749E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40276E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53960E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62850E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50363E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49454E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20340E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.13037E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333656 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33373E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33373E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.63517E+03 ;
RUNNING_TIME              (idx, 1)        =  2.78922E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.07133E-01  1.29417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.76917E+02  5.15208E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.24500E-02  1.83667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.54050E+01  4.29002E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.74670E+02  7.37646E+02 ];
CPU_USAGE                 (idx, 1)        = 9.44769 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99017E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.39857E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.62849E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67901E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.16261E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97254E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37464E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62381E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50939E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10175E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.80623E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.73746E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.52167E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.26522E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.95377E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35648E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.34678E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33389E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54293E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.03264E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17762E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14175E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55869E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.33928E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.87095E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.52673E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.38905E-03 -7.48292E+26  2.21546E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.63078E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.61638E+18 0.00424  1.72349E-02 0.00419 ];
U233_FISS                 (idx, [1:   4]) = [  9.12221E+19 0.00053  9.72686E-01 9.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.33304E+17 0.00849  3.55415E-03 0.00849 ];
TH232_CAPT                (idx, [1:   4]) = [  9.92617E+19 0.00049  7.37565E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33302E+19 0.00144  9.90486E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  1.17866E+17 0.01457  8.75893E-04 0.01460 ];
SM149_CAPT                (idx, [1:   4]) = [  1.51897E+17 0.01314  1.12870E-03 0.01313 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001191 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.18833E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001191 1.00119E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5358306 5.36413E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3734014 3.73804E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908871 9.09714E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001191 1.00119E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.12342E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36406E+20 3.7E-06  2.36406E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37701E+19 1.1E-07  9.37701E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34571E+20 0.00029  1.20571E+20 0.00020  1.39996E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.28341E+20 0.00017  2.14342E+20 0.00011  1.39996E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.50891E+20 0.00033  2.50891E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32996E+22 0.00023  3.16030E+22 0.00021  1.69665E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.28245E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.51166E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12891E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50749E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50749E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.66341E-01 0.05943 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27494E-01 0.01376 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.56110E-04 0.02770 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.06070E+03 0.03048 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09030E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.25757E-01 0.04690 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.68967E-01 0.04690 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52113E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.42318E-01 0.00046  1.04373E-01 0.00044  3.38088E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.42332E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.42296E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.42332E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03664E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02299E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02146E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33708E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33894E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.55940E-01 0.00174 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.55954E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.45876E-03 0.00520  2.90483E-04 0.01796  8.40961E-04 0.01057  6.31732E-04 0.01295  1.35668E-03 0.00843  2.89266E-04 0.01987  4.96379E-05 0.04734 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.37447E-01 0.01456  1.24757E-02 2.8E-05  3.23292E-02 9.9E-05  1.06142E-01 0.00059  2.97847E-01 0.00028  1.23766E+00 0.00035  4.81791E+00 0.04128 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24742E-03 0.00727  2.79895E-04 0.02687  7.93225E-04 0.01622  6.01573E-04 0.01919  1.25249E-03 0.01241  2.73900E-04 0.02910  4.63358E-05 0.06702 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.36192E-01 0.02017  1.24756E-02 3.9E-05  3.23247E-02 0.00014  1.06176E-01 0.00085  2.97748E-01 0.00042  1.23714E+00 0.00045  6.10304E+00 0.03051 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68632E-07 0.00131  8.68245E-07 0.00131  9.87052E-07 0.01962 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.18465E-07 0.00119  8.18101E-07 0.00120  9.30032E-07 0.01965 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23094E-03 0.00918  2.84808E-04 0.03198  7.78736E-04 0.01922  5.89842E-04 0.02133  1.25871E-03 0.01451  2.75800E-04 0.02959  4.30394E-05 0.08413 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.29908E-01 0.02381  1.24753E-02 5.3E-05  3.23321E-02 0.00017  1.06091E-01 0.00097  2.97919E-01 0.00049  1.23697E+00 0.00053  6.03474E+00 0.04672 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.70173E-07 0.01969  7.69988E-07 0.01969  7.98803E-07 0.05138 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.25993E-07 0.01969  7.25819E-07 0.01969  7.52929E-07 0.05138 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.75882E-03 0.04288  2.09621E-04 0.14089  6.40681E-04 0.07742  4.97089E-04 0.09244  1.12979E-03 0.06539  2.46587E-04 0.13630  3.50524E-05 0.32307 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.38446E-01 0.10330  1.24753E-02 0.00016  3.23501E-02 0.00062  1.06234E-01 0.00348  2.98142E-01 0.00173  1.23762E+00 0.00150  6.90121E+00 0.14089 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.76884E-03 0.04214  2.11252E-04 0.14164  6.24570E-04 0.07733  5.06805E-04 0.09105  1.14747E-03 0.06341  2.44389E-04 0.12794  3.43562E-05 0.31103 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.40245E-01 0.10208  1.24753E-02 0.00016  3.23515E-02 0.00063  1.06211E-01 0.00345  2.98102E-01 0.00170  1.23756E+00 0.00151  6.90121E+00 0.14089 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.60422E+03 0.03857 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.63444E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.13592E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19239E-03 0.00599 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.69744E+03 0.00598 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77819E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.61622E-05 0.00939  6.61770E-05 0.00940  6.24719E-07 0.59021 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.84765E-05 0.03117  5.84104E-05 0.03119  7.46236E-07 0.69017 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52132E-04 0.02628  1.52358E-04 0.02638  8.08779E-05 0.57698 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74031E+01 0.01089 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49454E+01 0.00024  4.05114E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 20:57:33 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01481E+00  9.99700E-01  9.96093E-01  1.00352E+00  9.96911E-01  1.00191E+00  1.00158E+00  1.00351E+00  1.00366E+00  9.97034E-01  9.88953E-01  9.96382E-01  9.97333E-01  9.89271E-01  1.00393E+00  1.00430E+00  1.00165E+00  9.99445E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.62396E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63760E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40171E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53852E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62734E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51971E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51067E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23672E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.13755E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334313 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33391E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33391E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.12243E+03 ;
RUNNING_TIME              (idx, 1)        =  3.31729E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.42183E-01  1.35050E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.29551E+02  5.26345E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.11550E-01  1.91000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.97973E+01  4.39230E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.27377E+02  7.40777E+02 ];
CPU_USAGE                 (idx, 1)        = 9.41258 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99027E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.35926E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.65649E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67829E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.13582E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99500E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38755E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62896E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50700E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27301E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.94125E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.72875E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.65435E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.95564E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.07491E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07968E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32410E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31572E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51825E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.85920E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.79608E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14715E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.57193E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.91056E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.91159E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.62305E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43284E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.71525E-03 -1.26191E+27  2.22059E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.89457E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.62465E+18 0.00402  1.73279E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  9.03563E+19 0.00057  9.63729E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  8.09468E+17 0.00513  8.63441E-03 0.00515 ];
PU239_FISS                (idx, [1:   4]) = [  5.09204E+13 0.70596  5.50082E-07 0.70601 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00615E+20 0.00052  7.30994E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31955E+19 0.00138  9.58689E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  2.93286E+17 0.00967  2.13075E-03 0.00966 ];
PU239_CAPT                (idx, [1:   4]) = [  2.56391E+13 1.00000  1.85773E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.65904E+17 0.01250  1.20553E-03 0.01253 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001741 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14392E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001741 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5411337 5.41683E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3686350 3.68976E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904054 9.04847E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001741 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.35276E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36373E+20 4.1E-06  2.36373E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37628E+19 1.3E-07  9.37628E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37691E+20 0.00028  1.23329E+20 0.00020  1.43626E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31454E+20 0.00017  2.17092E+20 0.00012  1.43626E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.54102E+20 0.00038  2.54102E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.38157E+22 0.00026  3.20805E+22 0.00023  1.73511E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.29933E+19 0.00157 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.54448E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14741E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49931E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49931E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.68345E-01 0.05973 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.08862E-01 0.01603 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.48010E-04 0.02767 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.26649E+03 0.02910 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09517E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.23633E-01 0.04624 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.67259E-01 0.04624 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52097E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.29968E-01 0.00046  1.03015E-01 0.00045  3.37442E-04 0.00906 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.30051E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.30270E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.30051E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02256E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01375E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01445E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34958E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34838E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.60213E-01 0.00163 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.60315E-01 0.00052 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.56113E-03 0.00596  2.79705E-04 0.01922  8.72835E-04 0.01130  6.40687E-04 0.01265  1.40659E-03 0.00918  2.97687E-04 0.01923  6.36208E-05 0.04090 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.63142E-01 0.01428  1.24754E-02 2.8E-05  3.23210E-02 9.7E-05  1.06266E-01 0.00061  2.98062E-01 0.00028  1.23921E+00 0.00040  5.63354E+00 0.03354 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27098E-03 0.00814  2.56382E-04 0.02915  8.05674E-04 0.01671  5.87373E-04 0.01812  1.28498E-03 0.01362  2.79939E-04 0.02696  5.66381E-05 0.06106 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.62108E-01 0.02060  1.24759E-02 3.5E-05  3.23163E-02 0.00013  1.06240E-01 0.00086  2.97932E-01 0.00035  1.23937E+00 0.00048  6.53276E+00 0.02658 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.79038E-07 0.00130  8.78607E-07 0.00130  1.01108E-06 0.02241 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17418E-07 0.00119  8.17017E-07 0.00118  9.40303E-07 0.02235 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26775E-03 0.00939  2.61537E-04 0.03298  8.00445E-04 0.01935  5.82641E-04 0.02085  1.28111E-03 0.01347  2.81152E-04 0.03105  6.08629E-05 0.06410 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.69904E-01 0.02333  1.24757E-02 5.1E-05  3.23163E-02 0.00017  1.06202E-01 0.00095  2.97891E-01 0.00048  1.23815E+00 0.00061  6.53158E+00 0.03726 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.74263E-07 0.01961  7.73878E-07 0.01961  9.06733E-07 0.06141 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20145E-07 0.01959  7.19788E-07 0.01959  8.43321E-07 0.06138 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.90868E-03 0.04447  1.60244E-04 0.17487  7.69014E-04 0.08132  4.70475E-04 0.09499  1.20117E-03 0.06639  2.75558E-04 0.13667  3.22176E-05 0.26813 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.20199E-01 0.06393  1.24734E-02 0.00019  3.22970E-02 0.00053  1.06056E-01 0.00329  2.98465E-01 0.00180  1.23705E+00 0.00150  5.78010E+00 0.12389 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91762E-03 0.04401  1.61213E-04 0.17321  7.73866E-04 0.08019  4.73016E-04 0.09332  1.19901E-03 0.06680  2.80802E-04 0.13278  2.97138E-05 0.26877 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.20433E-01 0.06206  1.24734E-02 0.00020  3.22982E-02 0.00054  1.06074E-01 0.00328  2.98431E-01 0.00178  1.23689E+00 0.00149  5.78010E+00 0.12389 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.76997E+03 0.04005 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.71431E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10351E-07 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25750E-03 0.00542 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.73885E+03 0.00550 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.79843E-09 0.00069 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63559E-05 0.01044  6.63131E-05 0.01042  1.54701E-06 0.45124 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.98008E-05 0.03029  5.97871E-05 0.03028  1.03046E-06 0.50017 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.44419E-04 0.02646  1.44434E-04 0.02651  1.28956E-04 0.44460 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69404E+01 0.01104 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51067E+01 0.00026  4.04942E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 21:51:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01575E+00  1.00160E+00  9.95445E-01  9.94147E-01  1.00109E+00  9.97484E-01  9.93588E-01  1.00087E+00  9.95171E-01  9.99066E-01  9.93362E-01  9.93382E-01  1.00924E+00  1.00419E+00  1.00240E+00  1.00374E+00  1.00109E+00  9.98380E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.62934E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63707E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39936E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53640E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62581E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51844E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50938E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24198E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.14191E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333635 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33368E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33368E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.61730E+03 ;
RUNNING_TIME              (idx, 1)        =  3.85330E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.07738E+00  1.35200E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.82975E+02  5.34234E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.30533E-01  1.89833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.42202E+01  4.42280E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.80947E+02  7.50468E+02 ];
CPU_USAGE                 (idx, 1)        = 9.38752 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99030E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.33123E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.73198E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69164E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.99471E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01826E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39816E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.68064E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51881E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75477E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.24233E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.12868E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.88225E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.25661E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.35091E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.33210E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.32330E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31195E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51464E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.59514E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.28226E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.18192E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.67693E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84529E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98797E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.73035E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.15693E-02 -2.55447E+27  2.23352E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01905E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.61990E+18 0.00393  1.72845E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  8.81000E+19 0.00054  9.40026E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.28686E+18 0.00335  2.44012E-02 0.00333 ];
PU239_FISS                (idx, [1:   4]) = [  6.44768E+14 0.20063  6.88444E-06 0.20067 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01002E+20 0.00053  7.16974E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28945E+19 0.00143  9.15351E-02 0.00140 ];
U235_CAPT                 (idx, [1:   4]) = [  8.07699E+17 0.00574  5.73338E-03 0.00571 ];
PU239_CAPT                (idx, [1:   4]) = [  4.12605E+14 0.25893  2.92737E-06 0.25931 ];
PU240_CAPT                (idx, [1:   4]) = [  2.60772E+13 1.00000  1.86595E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.65933E+17 0.01196  1.17792E-03 0.01197 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001046 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13496E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001046 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5461206 5.46700E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3633445 3.63715E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906395 9.07200E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001046 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.17233E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36281E+20 4.2E-06  2.36281E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37410E+19 1.2E-07  9.37410E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40831E+20 0.00030  1.26286E+20 0.00020  1.45447E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34572E+20 0.00018  2.20027E+20 0.00012  1.45447E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57678E+20 0.00036  2.57678E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43029E+22 0.00026  3.25466E+22 0.00023  1.75633E+21 0.00189 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.33771E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.57949E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16321E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47796E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47796E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.75961E-01 0.05582 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.34547E-01 0.01370 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.54253E-04 0.02572 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.46096E+03 0.02544 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09282E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.68674E-01 0.04120 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.08038E-01 0.04120 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52057E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99747E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.16793E-01 0.00043  1.01528E-01 0.00043  3.34542E-04 0.00844 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.17056E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.16996E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.17056E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00856E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.98416E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.98381E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.39007E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.39034E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.69811E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.70068E-01 0.00052 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.66922E-03 0.00523  2.84314E-04 0.01916  8.69671E-04 0.01105  6.84975E-04 0.01226  1.44831E-03 0.00847  3.16819E-04 0.01928  6.51255E-05 0.04275 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.67662E-01 0.01502  1.24751E-02 3.1E-05  3.22955E-02 0.00012  1.06416E-01 0.00056  2.98820E-01 0.00029  1.24382E+00 0.00049  5.49190E+00 0.03473 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.34632E-03 0.00801  2.60784E-04 0.02889  7.81588E-04 0.01681  6.28246E-04 0.01781  1.32556E-03 0.01263  2.88632E-04 0.02857  6.15152E-05 0.05888 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.76313E-01 0.02228  1.24748E-02 4.3E-05  3.22955E-02 0.00016  1.06461E-01 0.00080  2.98686E-01 0.00039  1.24315E+00 0.00062  6.51839E+00 0.02637 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82447E-07 0.00121  8.82028E-07 0.00121  1.00757E-06 0.01684 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.08979E-07 0.00114  8.08595E-07 0.00114  9.23745E-07 0.01685 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28683E-03 0.00858  2.53187E-04 0.03117  7.75056E-04 0.02033  6.10309E-04 0.02003  1.30409E-03 0.01399  2.85630E-04 0.03012  5.85613E-05 0.07056 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.76663E-01 0.02764  1.24766E-02 4.3E-05  3.23004E-02 0.00022  1.06392E-01 0.00108  2.98620E-01 0.00047  1.24390E+00 0.00096  6.49851E+00 0.03653 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.79298E-07 0.01959  7.78764E-07 0.01959  9.23290E-07 0.05334 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.14534E-07 0.01959  7.14045E-07 0.01959  8.46306E-07 0.05332 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.90214E-03 0.04381  2.28667E-04 0.12438  6.74562E-04 0.08753  5.58663E-04 0.09494  1.13498E-03 0.06383  2.69963E-04 0.13242  3.53075E-05 0.29882 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.45091E-01 0.08084  1.24772E-02 0.00012  3.22643E-02 0.00070  1.06455E-01 0.00342  2.99234E-01 0.00189  1.24616E+00 0.00270  6.07305E+00 0.11560 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87254E-03 0.04306  2.26050E-04 0.12128  6.56737E-04 0.08486  5.56442E-04 0.09414  1.13746E-03 0.06251  2.58750E-04 0.12950  3.70952E-05 0.28257 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.46802E-01 0.08259  1.24772E-02 0.00012  3.22615E-02 0.00069  1.06514E-01 0.00343  2.99259E-01 0.00188  1.24623E+00 0.00269  6.07305E+00 0.11560 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.72314E+03 0.03872 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.75798E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.02883E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.33848E-03 0.00665 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81287E+03 0.00674 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.80365E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70783E-05 0.00905  6.71102E-05 0.00906  9.14393E-07 0.45063 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88751E-05 0.02955  5.87516E-05 0.02972  1.66788E-06 0.56085 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52824E-04 0.02459  1.52886E-04 0.02462  1.40719E-04 0.44500 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67773E+01 0.01166 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50938E+01 0.00026  4.02636E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 22:45:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01281E+00  9.96737E-01  1.00676E+00  1.00377E+00  9.99575E-01  1.00041E+00  1.00472E+00  1.00345E+00  1.00199E+00  9.96429E-01  9.90440E-01  9.99070E-01  9.96512E-01  9.95287E-01  9.96728E-01  1.00041E+00  9.98796E-01  9.96100E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.64249E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63575E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39708E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53459E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62539E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49503E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48595E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20571E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.14161E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333085 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33374E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33374E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.11785E+03 ;
RUNNING_TIME              (idx, 1)        =  4.39484E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.21652E+00  1.39133E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.36945E+02  5.39697E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.48733E-01  1.82000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.86217E+01  4.40143E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.35123E+02  7.56608E+02 ];
CPU_USAGE                 (idx, 1)        = 9.36975 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99055E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.31134E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.87825E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73166E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.50232E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01933E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39300E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82516E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55875E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.10207E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77926E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.01976E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20511E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04704E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85179E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38547E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38453E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35314E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57416E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.87855E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30918E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26667E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91641E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.42491E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10780E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77418E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.20128E-02 -4.86037E+27  2.25658E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02802E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58318E+18 0.00414  1.68911E-02 0.00409 ];
U233_FISS                 (idx, [1:   4]) = [  8.47569E+19 0.00055  9.04300E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.64019E+18 0.00235  4.95079E-02 0.00230 ];
PU239_FISS                (idx, [1:   4]) = [  2.00376E+16 0.03517  2.13871E-04 0.03524 ];
PU240_FISS                (idx, [1:   4]) = [  5.68516E+14 0.21554  6.05257E-06 0.21544 ];
PU241_FISS                (idx, [1:   4]) = [  4.67527E+14 0.24216  4.97677E-06 0.24217 ];
TH232_CAPT                (idx, [1:   4]) = [  9.88089E+19 0.00051  6.95090E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23769E+19 0.00144  8.70661E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  1.64795E+18 0.00389  1.15930E-02 0.00388 ];
U238_CAPT                 (idx, [1:   4]) = [  1.03200E+14 0.49754  7.24928E-07 0.49752 ];
PU239_CAPT                (idx, [1:   4]) = [  8.63566E+15 0.05121  6.07312E-05 0.05120 ];
PU240_CAPT                (idx, [1:   4]) = [  1.63083E+15 0.13087  1.14807E-05 0.13097 ];
PU241_CAPT                (idx, [1:   4]) = [  1.82427E+14 0.37421  1.27834E-06 0.37418 ];
SM149_CAPT                (idx, [1:   4]) = [  1.70767E+17 0.01265  1.20128E-03 0.01265 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001228 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09384E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001228 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5480044 5.48562E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3613459 3.61686E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907725 9.08463E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001228 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.22935E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36157E+20 4.3E-06  2.36157E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37032E+19 1.1E-07  9.37032E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42012E+20 0.00028  1.27531E+20 0.00020  1.44817E+19 0.00196 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35716E+20 0.00017  2.21234E+20 0.00012  1.44817E+19 0.00196 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59139E+20 0.00035  2.59139E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43883E+22 0.00025  3.26406E+22 0.00022  1.74768E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35426E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59258E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16370E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42421E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42421E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.32360E-01 0.05722 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.25327E-01 0.01420 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.48760E-04 0.02605 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.90062E+03 0.02927 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09157E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.31455E-01 0.04432 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.74040E-01 0.04432 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52026E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99828E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11492E-01 0.00044  1.00945E-01 0.00043  3.40240E-04 0.00912 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11916E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11345E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11916E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00310E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92385E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92440E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47649E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47542E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83824E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83787E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.77729E-03 0.00529  2.85052E-04 0.01952  9.00168E-04 0.01102  6.81430E-04 0.01214  1.49947E-03 0.00787  3.38305E-04 0.01869  7.28665E-05 0.03764 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.90024E-01 0.01384  1.24757E-02 3.1E-05  3.22700E-02 0.00013  1.06488E-01 0.00056  3.00138E-01 0.00030  1.25403E+00 0.00069  6.20011E+00 0.02762 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.41646E-03 0.00753  2.51712E-04 0.02925  8.30115E-04 0.01588  6.08030E-04 0.01757  1.36602E-03 0.01247  2.96693E-04 0.02681  6.38921E-05 0.05770 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.85735E-01 0.02193  1.24755E-02 4.1E-05  3.22634E-02 0.00019  1.06493E-01 0.00075  3.00180E-01 0.00040  1.25422E+00 0.00096  6.81822E+00 0.02320 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.63974E-07 0.00126  8.63525E-07 0.00127  9.98493E-07 0.01654 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.87460E-07 0.00119  7.87051E-07 0.00119  9.10074E-07 0.01654 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.36593E-03 0.00917  2.57052E-04 0.03332  8.12677E-04 0.01966  6.04538E-04 0.02074  1.32865E-03 0.01319  2.96748E-04 0.03156  6.62688E-05 0.06372 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.86260E-01 0.02416  1.24754E-02 5.8E-05  3.22686E-02 0.00023  1.06473E-01 0.00101  3.00129E-01 0.00051  1.25060E+00 0.00104  6.63856E+00 0.03409 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.62609E-07 0.01969  7.62558E-07 0.01969  8.15650E-07 0.05716 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.95193E-07 0.01968  6.95145E-07 0.01969  7.43754E-07 0.05718 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.88416E-03 0.04349  2.39370E-04 0.12964  7.35996E-04 0.07667  5.25071E-04 0.09787  1.08457E-03 0.06261  2.37916E-04 0.13629  6.12320E-05 0.28128 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.05018E-01 0.12862  1.24755E-02 0.00015  3.22838E-02 0.00075  1.06552E-01 0.00286  3.00497E-01 0.00201  1.25198E+00 0.00384  6.64682E+00 0.10440 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90460E-03 0.04303  2.41921E-04 0.12538  7.45146E-04 0.07736  5.30626E-04 0.09498  1.09153E-03 0.06227  2.36317E-04 0.13209  5.90596E-05 0.27775 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.03306E-01 0.12745  1.24755E-02 0.00015  3.22842E-02 0.00076  1.06584E-01 0.00287  3.00519E-01 0.00201  1.25180E+00 0.00383  6.64704E+00 0.10440 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79094E+03 0.03888 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.57210E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.81298E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34938E-03 0.00558 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90779E+03 0.00561 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76086E-09 0.00068 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.55936E-05 0.01061  6.55373E-05 0.01063  1.19279E-06 0.45425 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.69483E-05 0.03139  5.69831E-05 0.03146  8.05557E-07 0.59296 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.48328E-04 0.02556  1.48386E-04 0.02572  1.29837E-04 0.44570 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63401E+01 0.01089 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48595E+01 0.00026  3.97033E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Apr 28 23:39:20 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00720E+00  1.00083E+00  1.00262E+00  1.00068E+00  9.95103E-01  1.00477E+00  9.99233E-01  9.88003E-01  9.99748E-01  1.00230E+00  9.96700E-01  9.97950E-01  1.00473E+00  9.98376E-01  1.00747E+00  1.00158E+00  9.95436E-01  9.97274E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.64912E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63509E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39860E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53621E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62589E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49255E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48350E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19538E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.14548E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333552 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33364E+04 0.00052 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33364E+04 0.00052 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.61770E+03 ;
RUNNING_TIME              (idx, 1)        =  4.93511E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.36293E+00  1.46417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.90777E+02  5.38321E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.67983E-01  1.92500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.29509E+01  4.32920E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.89225E+02  7.59902E+02 ];
CPU_USAGE                 (idx, 1)        = 9.35685 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99043E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.29699E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.90004E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73763E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.46682E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01121E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38665E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85483E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56513E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.02271E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98862E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.84709E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.37294E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13544E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04341E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.82158E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38523E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35228E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57372E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92550E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73892E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27432E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96739E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.83404E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11512E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.76834E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.71182E-02 -5.98762E+27  2.26785E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02553E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.56591E+18 0.00410  1.67130E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  8.36405E+19 0.00050  8.92732E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.35770E+18 0.00217  5.71838E-02 0.00209 ];
U238_FISS                 (idx, [1:   4]) = [  2.63145E+13 1.00000  2.77639E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  4.61496E+16 0.02185  4.92545E-04 0.02182 ];
PU240_FISS                (idx, [1:   4]) = [  1.60409E+15 0.13258  1.70792E-05 0.13239 ];
PU241_FISS                (idx, [1:   4]) = [  1.94212E+15 0.11520  2.07375E-05 0.11498 ];
TH232_CAPT                (idx, [1:   4]) = [  9.75210E+19 0.00051  6.86407E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22084E+19 0.00152  8.59297E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  1.88278E+18 0.00366  1.32525E-02 0.00366 ];
U238_CAPT                 (idx, [1:   4]) = [  3.64004E+14 0.26139  2.55020E-06 0.26140 ];
PU239_CAPT                (idx, [1:   4]) = [  2.15920E+16 0.03399  1.51979E-04 0.03396 ];
PU240_CAPT                (idx, [1:   4]) = [  4.71071E+15 0.07371  3.31609E-05 0.07371 ];
PU241_CAPT                (idx, [1:   4]) = [  3.89309E+14 0.25210  2.74759E-06 0.25209 ];
SM149_CAPT                (idx, [1:   4]) = [  1.70769E+17 0.01274  1.20213E-03 0.01278 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000921 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10617E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000921 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5480978 5.48670E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3614616 3.61820E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905327 9.06156E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000921 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.11759E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36143E+20 4.7E-06  2.36143E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36890E+19 1.1E-07  9.36890E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42158E+20 0.00030  1.27560E+20 0.00021  1.45977E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35847E+20 0.00018  2.21249E+20 0.00012  1.45977E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58945E+20 0.00035  2.58945E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43454E+22 0.00023  3.25829E+22 0.00021  1.76246E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.34654E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59312E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16218E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39639E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39639E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.63671E-01 0.05480 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27566E-01 0.01283 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.56276E-04 0.02589 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.12194E+03 0.02765 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09387E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.51694E-01 0.04244 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.92702E-01 0.04244 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52050E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99858E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11836E-01 0.00039  1.00985E-01 0.00039  3.45788E-04 0.00848 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11674E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11978E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11674E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00249E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90383E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90364E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50641E-02 0.00149 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50637E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.88173E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.88410E-01 0.00051 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.87154E-03 0.00510  2.81436E-04 0.02083  9.10053E-04 0.01013  6.85453E-04 0.01269  1.56681E-03 0.00817  3.50568E-04 0.01792  7.72147E-05 0.03841 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.04401E-01 0.01428  1.24765E-02 3.9E-05  3.22509E-02 0.00013  1.06469E-01 0.00054  3.00367E-01 0.00030  1.25685E+00 0.00067  6.46205E+00 0.02747 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43924E-03 0.00714  2.54073E-04 0.02913  7.97752E-04 0.01503  6.00658E-04 0.01765  1.40084E-03 0.01181  3.16510E-04 0.02574  6.94130E-05 0.06058 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.08459E-01 0.02301  1.24761E-02 4.2E-05  3.22446E-02 0.00018  1.06515E-01 0.00084  3.00492E-01 0.00043  1.25713E+00 0.00099  7.16277E+00 0.02153 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.54999E-07 0.00130  8.54588E-07 0.00130  9.75965E-07 0.01652 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.79583E-07 0.00124  7.79208E-07 0.00124  8.89910E-07 0.01652 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41991E-03 0.00896  2.50420E-04 0.03463  7.84465E-04 0.01958  5.95464E-04 0.02066  1.40374E-03 0.01352  3.12249E-04 0.02934  7.35690E-05 0.06719 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.18924E-01 0.02700  1.24782E-02 0.00011  3.22381E-02 0.00025  1.06500E-01 0.00101  3.00126E-01 0.00047  1.25645E+00 0.00135  7.14988E+00 0.02970 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.60536E-07 0.01967  7.60121E-07 0.01967  8.80077E-07 0.05778 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.93585E-07 0.01966  6.93206E-07 0.01967  8.02634E-07 0.05781 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08126E-03 0.04393  2.88517E-04 0.13107  7.32528E-04 0.07635  5.45275E-04 0.09276  1.21090E-03 0.06586  2.55814E-04 0.15628  4.82300E-05 0.26526 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.46555E-01 0.09479  1.24774E-02 0.00012  3.22077E-02 0.00069  1.06615E-01 0.00317  2.99354E-01 0.00167  1.25703E+00 0.00393  7.26281E+00 0.10061 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.07017E-03 0.04337  2.80329E-04 0.12875  7.36939E-04 0.07582  5.46611E-04 0.09274  1.20130E-03 0.06457  2.55984E-04 0.14238  4.90070E-05 0.26854 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.54010E-01 0.09774  1.24774E-02 0.00012  3.22084E-02 0.00068  1.06586E-01 0.00314  2.99376E-01 0.00166  1.25696E+00 0.00391  7.26281E+00 0.10061 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.07378E+03 0.03967 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.49456E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.74527E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43777E-03 0.00591 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.04748E+03 0.00593 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.75288E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73213E-05 0.00904  6.73466E-05 0.00905  6.04465E-07 0.58468 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65303E-05 0.02997  5.65281E-05 0.02996  7.81270E-07 0.64378 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.55098E-04 0.02471  1.55400E-04 0.02479  7.42839E-05 0.57685 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62126E+01 0.01118 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48350E+01 0.00028  3.95295E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 00:34:27 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00971E+00  1.00026E+00  9.99778E-01  1.00238E+00  9.95348E-01  9.97774E-01  9.96955E-01  1.00085E+00  1.00671E+00  1.00471E+00  9.96019E-01  9.95231E-01  1.00539E+00  9.99268E-01  9.98269E-01  1.00221E+00  9.92604E-01  9.96534E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.65599E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63440E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39493E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53286E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62428E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47857E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46948E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18174E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.14582E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333785 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33413E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33413E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.12736E+03 ;
RUNNING_TIME              (idx, 1)        =  5.48635E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.50672E+00  1.43783E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.45707E+02  5.49303E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.87000E-01  1.90167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.74077E+01  4.45670E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.44218E+02  7.60198E+02 ];
CPU_USAGE                 (idx, 1)        = 9.34568 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99032E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.28456E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.90143E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73850E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.07693E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99730E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37689E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87015E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56699E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.08114E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.16970E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.82560E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.54087E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21235E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20710E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20620E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37753E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34519E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56470E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.94812E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.12077E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26771E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99180E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.14275E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09701E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.76838E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.22735E-02 -7.12590E+27  2.27923E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02305E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.56004E+18 0.00409  1.66542E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  8.28113E+19 0.00055  8.84113E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.85610E+18 0.00206  6.25223E-02 0.00203 ];
PU239_FISS                (idx, [1:   4]) = [  8.26632E+16 0.01788  8.82726E-04 0.01792 ];
PU240_FISS                (idx, [1:   4]) = [  3.25882E+15 0.08806  3.48205E-05 0.08824 ];
PU241_FISS                (idx, [1:   4]) = [  3.70440E+15 0.07963  3.95382E-05 0.07959 ];
TH232_CAPT                (idx, [1:   4]) = [  9.65428E+19 0.00052  6.79865E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20981E+19 0.00137  8.51972E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  2.07855E+18 0.00361  1.46377E-02 0.00361 ];
U238_CAPT                 (idx, [1:   4]) = [  3.12185E+14 0.28333  2.19861E-06 0.28333 ];
PU239_CAPT                (idx, [1:   4]) = [  3.67793E+16 0.02728  2.58968E-04 0.02723 ];
PU240_CAPT                (idx, [1:   4]) = [  9.58634E+15 0.05215  6.75100E-05 0.05212 ];
PU241_CAPT                (idx, [1:   4]) = [  8.55084E+14 0.17005  6.01491E-06 0.17018 ];
SM149_CAPT                (idx, [1:   4]) = [  1.70592E+17 0.01211  1.20151E-03 0.01216 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002384 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10516E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002384 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5478921 5.48391E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3614234 3.61723E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909229 9.09913E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002384 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.38419E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36153E+20 4.5E-06  2.36153E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36765E+19 1.0E-07  9.36765E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.41996E+20 0.00029  1.27557E+20 0.00020  1.44393E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35673E+20 0.00017  2.21233E+20 0.00012  1.44393E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58946E+20 0.00033  2.58946E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.42982E+22 0.00024  3.25565E+22 0.00022  1.74167E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35626E+19 0.00155 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59235E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15859E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.36911E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.36911E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88676E-01 0.05335 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.38820E-01 0.01284 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.55254E-04 0.02671 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.88935E+03 0.02869 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09010E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.62303E-01 0.04151 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.02021E-01 0.04151 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52094E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99885E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11703E-01 0.00046  1.00975E-01 0.00045  3.43987E-04 0.00908 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11976E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.12006E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11976E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00326E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88609E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88557E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53333E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53384E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.91913E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.92301E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.82580E-03 0.00500  2.85747E-04 0.01899  9.00244E-04 0.00995  7.04917E-04 0.01275  1.51015E-03 0.00828  3.48795E-04 0.01767  7.59490E-05 0.04075 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.96131E-01 0.01513  1.24771E-02 3.6E-05  3.22465E-02 0.00012  1.06613E-01 0.00053  3.00548E-01 0.00030  1.25831E+00 0.00075  5.96819E+00 0.02979 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.40547E-03 0.00727  2.49696E-04 0.02831  7.99504E-04 0.01488  6.29237E-04 0.01838  1.34809E-03 0.01136  3.12214E-04 0.02657  6.67361E-05 0.05885 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.96908E-01 0.02216  1.24767E-02 3.6E-05  3.22465E-02 0.00017  1.06569E-01 0.00075  3.00515E-01 0.00045  1.25820E+00 0.00102  6.87827E+00 0.02253 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.49574E-07 0.00133  8.49257E-07 0.00132  9.45183E-07 0.01566 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.74508E-07 0.00124  7.74219E-07 0.00123  8.61619E-07 0.01564 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.38627E-03 0.00920  2.45294E-04 0.03480  7.94315E-04 0.01830  6.28915E-04 0.02193  1.34149E-03 0.01397  3.04332E-04 0.03051  7.19176E-05 0.06528 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.03046E-01 0.02503  1.24772E-02 7.7E-05  3.22420E-02 0.00021  1.06570E-01 0.00098  3.00611E-01 0.00051  1.25980E+00 0.00142  6.79621E+00 0.03149 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.53768E-07 0.01962  7.53377E-07 0.01962  8.53708E-07 0.05720 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.87419E-07 0.01962  6.87061E-07 0.01962  7.79026E-07 0.05725 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94651E-03 0.04299  1.99708E-04 0.15346  7.02992E-04 0.08648  5.43264E-04 0.09525  1.18909E-03 0.06217  2.67322E-04 0.11382  4.41397E-05 0.32532 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.78301E-01 0.10841  1.24776E-02 0.00014  3.22274E-02 0.00086  1.06849E-01 0.00362  3.00957E-01 0.00196  1.26238E+00 0.00404  5.97366E+00 0.11349 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94083E-03 0.04302  2.00568E-04 0.14911  7.04320E-04 0.08501  5.40104E-04 0.09531  1.18193E-03 0.06143  2.72248E-04 0.11304  4.16567E-05 0.30525 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.74395E-01 0.10043  1.24776E-02 0.00014  3.22294E-02 0.00084  1.06859E-01 0.00362  3.00932E-01 0.00193  1.26244E+00 0.00403  5.97292E+00 0.11348 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.93048E+03 0.03858 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.43261E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.68762E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40397E-03 0.00597 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03692E+03 0.00596 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.74958E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67522E-05 0.00969  6.67216E-05 0.00969  1.10353E-06 0.50495 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.75250E-05 0.02913  5.75758E-05 0.02910  3.29308E-07 0.50840 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52854E-04 0.02572  1.53039E-04 0.02576  1.11765E-04 0.50063 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65423E+01 0.01049 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46948E+01 0.00026  3.93397E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 01:30:24 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01296E+00  9.95606E-01  9.98331E-01  9.99193E-01  1.00027E+00  9.97615E-01  9.99708E-01  9.95871E-01  9.99247E-01  9.97076E-01  9.97228E-01  9.92553E-01  1.00138E+00  9.95788E-01  1.00835E+00  1.00483E+00  1.00866E+00  9.95332E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.67760E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63224E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39161E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53013E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62857E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46576E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45665E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16799E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.16004E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334493 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33426E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33426E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.64556E+03 ;
RUNNING_TIME              (idx, 1)        =  6.04575E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.65328E+00  1.46567E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.01448E+02  5.57405E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.05417E-01  1.84167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.18179E+01  4.41013E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.00204E+02  7.65490E+02 ];
CPU_USAGE                 (idx, 1)        = 9.33807 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99060E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.27611E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.92181E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74422E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.34638E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96246E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35602E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92538E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57483E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.59598E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.74786E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.10494E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03511E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44253E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70318E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34460E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37099E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33733E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55563E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.98780E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27294E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26286E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04784E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.40388E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07217E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.78885E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.28246E-02 -1.16635E+28  2.32461E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02153E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.53805E+18 0.00411  1.64262E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  8.07527E+19 0.00055  8.62472E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.86692E+18 0.00188  7.33434E-02 0.00185 ];
U238_FISS                 (idx, [1:   4]) = [  7.75427E+13 0.57542  8.27660E-07 0.57545 ];
PU239_FISS                (idx, [1:   4]) = [  2.87876E+17 0.00948  3.07460E-03 0.00945 ];
PU240_FISS                (idx, [1:   4]) = [  2.15337E+16 0.03439  2.29949E-04 0.03434 ];
PU241_FISS                (idx, [1:   4]) = [  3.11623E+16 0.02929  3.32965E-04 0.02933 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44228E+19 0.00052  6.62121E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17465E+19 0.00149  8.23729E-02 0.00151 ];
U235_CAPT                 (idx, [1:   4]) = [  2.40986E+18 0.00310  1.68990E-02 0.00310 ];
U238_CAPT                 (idx, [1:   4]) = [  1.11558E+15 0.14140  7.82988E-06 0.14139 ];
PU239_CAPT                (idx, [1:   4]) = [  1.37754E+17 0.01361  9.65933E-04 0.01359 ];
PU240_CAPT                (idx, [1:   4]) = [  5.40307E+16 0.02204  3.78774E-04 0.02200 ];
PU241_CAPT                (idx, [1:   4]) = [  6.43289E+15 0.05865  4.50525E-05 0.05854 ];
SM149_CAPT                (idx, [1:   4]) = [  1.76434E+17 0.01266  1.23719E-03 0.01264 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002765 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06606E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002765 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5488231 5.49275E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3603570 3.60630E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910964 9.11613E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002765 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.56348E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36299E+20 4.6E-06  2.36299E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36374E+19 8.9E-08  9.36374E+19 8.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42667E+20 0.00030  1.28063E+20 0.00020  1.46042E+19 0.00193 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36305E+20 0.00018  2.21700E+20 0.00011  1.46042E+19 0.00193 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59628E+20 0.00032  2.59628E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43814E+22 0.00023  3.26265E+22 0.00020  1.75483E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36690E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59974E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15826E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26564E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26564E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.28520E-01 0.05382 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.38196E-01 0.01217 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.75968E-04 0.02509 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.42980E+03 0.02424 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08841E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.00902E-01 0.03786 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.36990E-01 0.03786 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52356E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99968E+02 8.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09883E-01 0.00045  1.00774E-01 0.00045  3.45395E-04 0.00916 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09931E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.10171E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09931E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00118E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84211E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84391E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.60223E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59908E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.02668E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.02026E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.88851E-03 0.00529  2.74202E-04 0.02094  9.13051E-04 0.01015  6.92134E-04 0.01249  1.58312E-03 0.00878  3.48514E-04 0.01983  7.74966E-05 0.03871 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.01539E-01 0.01462  1.24782E-02 6.6E-05  3.22187E-02 0.00014  1.06716E-01 0.00053  3.01235E-01 0.00031  1.26060E+00 0.00092  6.23153E+00 0.02774 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44775E-03 0.00785  2.39031E-04 0.03048  8.05573E-04 0.01506  6.12514E-04 0.01823  1.39751E-03 0.01218  3.22876E-04 0.02721  7.02501E-05 0.05393 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.11556E-01 0.02193  1.24776E-02 6.2E-05  3.22214E-02 0.00019  1.06670E-01 0.00076  3.01144E-01 0.00043  1.25915E+00 0.00134  6.95694E+00 0.02170 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.35154E-07 0.00133  8.34775E-07 0.00134  9.48610E-07 0.01646 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.59842E-07 0.00124  7.59497E-07 0.00124  8.63044E-07 0.01644 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40718E-03 0.00928  2.41220E-04 0.03268  7.96433E-04 0.01897  6.03258E-04 0.02147  1.38051E-03 0.01572  3.18108E-04 0.03056  6.76542E-05 0.06278 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08524E-01 0.02478  1.24795E-02 0.00012  3.22130E-02 0.00023  1.06833E-01 0.00103  3.01200E-01 0.00055  1.26322E+00 0.00162  6.79802E+00 0.03231 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.38290E-07 0.01964  7.37938E-07 0.01963  8.17022E-07 0.05338 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.71835E-07 0.01963  6.71515E-07 0.01963  7.43317E-07 0.05328 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98830E-03 0.04473  2.52398E-04 0.12964  7.12637E-04 0.08747  5.73722E-04 0.08494  1.17977E-03 0.06906  2.20945E-04 0.13830  4.88254E-05 0.26786 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.39218E-01 0.08670  1.24907E-02 0.00115  3.22036E-02 0.00094  1.06870E-01 0.00307  3.01858E-01 0.00209  1.26425E+00 0.00414  6.75550E+00 0.10558 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00469E-03 0.04356  2.58370E-04 0.12701  6.98196E-04 0.08397  5.81785E-04 0.08386  1.20205E-03 0.06896  2.18878E-04 0.13536  4.54096E-05 0.26314 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.30140E-01 0.08326  1.24906E-02 0.00115  3.22010E-02 0.00094  1.06872E-01 0.00306  3.01847E-01 0.00208  1.26421E+00 0.00413  6.75550E+00 0.10558 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.06027E+03 0.04027 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.28568E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.53849E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43349E-03 0.00538 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.14466E+03 0.00545 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77481E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73089E-05 0.00780  6.73224E-05 0.00782  9.45427E-07 0.50648 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.72347E-05 0.02724  5.71132E-05 0.02734  1.40907E-06 0.56383 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.74317E-04 0.02403  1.74594E-04 0.02416  9.80722E-05 0.49834 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59551E+01 0.01245 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45665E+01 0.00026  3.89374E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 02:26:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01089E+00  1.00016E+00  1.00092E+00  1.00492E+00  9.97289E-01  1.00210E+00  9.93692E-01  9.92899E-01  1.00338E+00  1.00494E+00  9.96382E-01  9.94903E-01  9.97264E-01  9.99616E-01  9.99082E-01  1.00118E+00  1.00216E+00  9.98224E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.72616E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.62738E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37786E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51790E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64410E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46031E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45114E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20173E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.20562E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334107 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33376E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33376E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.16604E+03 ;
RUNNING_TIME              (idx, 1)        =  6.60741E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.80397E+00  1.50683E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.57407E+02  5.59592E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.24833E-01  1.94167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.62180E+01  4.40007E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.56380E+02  7.68201E+02 ];
CPU_USAGE                 (idx, 1)        = 9.33200 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99057E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26944E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.91063E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73907E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.14390E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92187E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33584E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95479E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57170E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53441E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38811E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.35979E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14036E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69222E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23710E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58050E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34464E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31609E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52723E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02026E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59724E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23201E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02822E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.13649E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01303E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.82143E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06463E-01 -2.35066E+28  2.44304E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02000E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.50793E+18 0.00414  1.61142E-02 0.00407 ];
U233_FISS                 (idx, [1:   4]) = [  7.93878E+19 0.00057  8.48392E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.09306E+18 0.00188  7.57998E-02 0.00177 ];
U238_FISS                 (idx, [1:   4]) = [  1.82500E+14 0.37419  1.95195E-06 0.37421 ];
PU239_FISS                (idx, [1:   4]) = [  6.07976E+17 0.00624  6.49702E-03 0.00620 ];
PU240_FISS                (idx, [1:   4]) = [  7.29668E+16 0.01873  7.79819E-04 0.01872 ];
PU241_FISS                (idx, [1:   4]) = [  1.19885E+17 0.01403  1.28115E-03 0.01400 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28486E+19 0.00055  6.47080E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15535E+19 0.00151  8.05202E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50001E+18 0.00324  1.74246E-02 0.00331 ];
U238_CAPT                 (idx, [1:   4]) = [  2.57347E+15 0.09926  1.79465E-05 0.09930 ];
PU239_CAPT                (idx, [1:   4]) = [  2.81152E+17 0.00969  1.95963E-03 0.00972 ];
PU240_CAPT                (idx, [1:   4]) = [  1.84179E+17 0.01263  1.28368E-03 0.01265 ];
PU241_CAPT                (idx, [1:   4]) = [  2.52749E+16 0.03376  1.76184E-04 0.03382 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78160E+17 0.01220  1.24152E-03 0.01217 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001271 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07449E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001271 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498323 5.50370E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3585895 3.58918E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 917053 9.17861E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001271 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.21072E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36606E+20 4.9E-06  2.36606E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35966E+19 1.1E-07  9.35966E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43385E+20 0.00029  1.28400E+20 0.00020  1.49852E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36982E+20 0.00017  2.21997E+20 0.00011  1.49852E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60714E+20 0.00036  2.60714E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47052E+22 0.00025  3.29167E+22 0.00022  1.78842E+21 0.00193 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39308E+19 0.00155 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60913E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16163E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.05758E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.05758E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.09792E-01 0.04955 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31399E-01 0.01187 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.06769E-04 0.02359 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.69092E+03 0.02783 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08218E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.29006E-01 0.03517 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.62094E-01 0.03517 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52794E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00056E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07425E-01 0.00045  1.00466E-01 0.00045  3.47924E-04 0.00945 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07830E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.07566E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07830E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99611E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81185E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81344E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65143E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64857E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09156E-01 0.00140 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.09335E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.92857E-03 0.00577  2.82574E-04 0.02001  9.09235E-04 0.01160  6.98162E-04 0.01154  1.59629E-03 0.00812  3.61041E-04 0.01748  8.12626E-05 0.03886 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.11860E-01 0.01461  1.24799E-02 0.00011  3.21973E-02 0.00016  1.06801E-01 0.00057  3.01373E-01 0.00028  1.25715E+00 0.00098  6.52007E+00 0.02612 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45940E-03 0.00816  2.50844E-04 0.03078  8.01015E-04 0.01568  6.09107E-04 0.01756  1.41139E-03 0.01200  3.08292E-04 0.02403  7.87574E-05 0.05520 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.14011E-01 0.02072  1.24810E-02 0.00018  3.21964E-02 0.00025  1.06879E-01 0.00081  3.01335E-01 0.00042  1.25641E+00 0.00120  7.01102E+00 0.02145 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.29396E-07 0.00130  8.29057E-07 0.00130  9.24640E-07 0.01746 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.52575E-07 0.00124  7.52267E-07 0.00124  8.39247E-07 0.01752 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44990E-03 0.00953  2.47249E-04 0.03478  7.80698E-04 0.01987  6.13428E-04 0.02198  1.42096E-03 0.01323  3.20057E-04 0.03001  6.75102E-05 0.06464 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.01608E-01 0.02327  1.24818E-02 0.00022  3.21909E-02 0.00029  1.06713E-01 0.00105  3.01549E-01 0.00048  1.25708E+00 0.00172  6.82374E+00 0.03235 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.32310E-07 0.01967  7.32057E-07 0.01967  8.04402E-07 0.05259 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.64653E-07 0.01967  6.64424E-07 0.01967  7.29684E-07 0.05253 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.24624E-03 0.04472  2.41007E-04 0.14025  6.89549E-04 0.09066  5.91379E-04 0.08157  1.34800E-03 0.06653  2.95206E-04 0.13991  8.10988E-05 0.23783 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.70088E-01 0.11071  1.24770E-02 0.00015  3.21477E-02 0.00094  1.07000E-01 0.00311  3.01739E-01 0.00196  1.25977E+00 0.00416  7.04607E+00 0.08423 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.25515E-03 0.04421  2.38023E-04 0.13688  6.84806E-04 0.08847  5.93931E-04 0.07869  1.36498E-03 0.06706  2.96253E-04 0.13613  7.71561E-05 0.22904 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.63346E-01 0.10929  1.24769E-02 0.00015  3.21475E-02 0.00094  1.07015E-01 0.00311  3.01748E-01 0.00195  1.25936E+00 0.00414  7.04607E+00 0.08423 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.45580E+03 0.04064 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.23399E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.47125E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.54996E-03 0.00548 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.31272E+03 0.00564 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.84877E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84001E-05 0.00687  6.83981E-05 0.00687  1.26880E-06 0.46222 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.84571E-05 0.02426  5.84503E-05 0.02429  1.63638E-06 0.49053 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.04997E-04 0.02227  2.05303E-04 0.02225  1.30474E-04 0.44470 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60646E+01 0.01057 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45114E+01 0.00028  3.86276E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 03:22:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00997E+00  1.00264E+00  9.99878E-01  9.97688E-01  9.95967E-01  9.97810E-01  9.99373E-01  1.00192E+00  9.98859E-01  9.97139E-01  9.91581E-01  9.96139E-01  1.00387E+00  1.00183E+00  1.00296E+00  1.00164E+00  9.93277E-01  1.00746E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.78094E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.62191E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36191E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50371E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65854E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46731E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45807E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26584E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.26323E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333771 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33401E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33401E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.68856E+03 ;
RUNNING_TIME              (idx, 1)        =  7.17156E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.96057E+00  1.56600E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.13605E+02  5.61982E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.44000E-01  1.91667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.06652E+01  4.44715E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.12757E+02  7.68892E+02 ];
CPU_USAGE                 (idx, 1)        = 9.32650 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99063E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26336E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.90897E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73752E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.36263E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91173E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33102E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96311E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57047E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64065E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56457E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45845E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.15993E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76426E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39329E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93051E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33744E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31078E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51982E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.03979E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00715E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22303E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01798E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34238E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99750E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.83149E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61232E-01 -3.55997E+28  2.56397E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01527E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.50034E+18 0.00414  1.60333E-02 0.00412 ];
U233_FISS                 (idx, [1:   4]) = [  7.92505E+19 0.00057  8.46902E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.08914E+18 0.00201  7.57585E-02 0.00197 ];
U238_FISS                 (idx, [1:   4]) = [  2.86518E+14 0.29644  3.04549E-06 0.29646 ];
PU239_FISS                (idx, [1:   4]) = [  6.56846E+17 0.00625  7.01978E-03 0.00627 ];
PU240_FISS                (idx, [1:   4]) = [  8.52759E+16 0.01861  9.11470E-04 0.01862 ];
PU241_FISS                (idx, [1:   4]) = [  1.46744E+17 0.01236  1.56816E-03 0.01234 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22471E+19 0.00054  6.42298E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15449E+19 0.00146  8.03870E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48256E+18 0.00339  1.72849E-02 0.00332 ];
U238_CAPT                 (idx, [1:   4]) = [  2.68588E+15 0.10453  1.87058E-05 0.10447 ];
PU239_CAPT                (idx, [1:   4]) = [  3.04560E+17 0.00936  2.12061E-03 0.00935 ];
PU240_CAPT                (idx, [1:   4]) = [  2.19353E+17 0.01122  1.52727E-03 0.01121 ];
PU241_CAPT                (idx, [1:   4]) = [  3.22660E+16 0.02838  2.24593E-04 0.02835 ];
SM149_CAPT                (idx, [1:   4]) = [  1.79466E+17 0.01210  1.24976E-03 0.01214 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002028 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07532E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002028 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496676 5.50166E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3581654 3.58467E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 923698 9.24423E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002028 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.72529E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36673E+20 4.6E-06  2.36673E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35899E+19 9.2E-08  9.35899E+19 9.2E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43612E+20 0.00029  1.28189E+20 0.00019  1.54223E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37202E+20 0.00018  2.21779E+20 0.00011  1.54223E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61050E+20 0.00033  2.61050E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50185E+22 0.00024  3.32014E+22 0.00021  1.81712E+21 0.00191 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41329E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61335E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16496E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87615E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87615E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.76022E-01 0.04876 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17540E-01 0.01128 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.54425E-04 0.02049 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.15925E+03 0.02937 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07562E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.65845E-01 0.03191 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.95099E-01 0.03191 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52883E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 9.2E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.06564E-01 0.00045  1.00376E-01 0.00044  3.48270E-04 0.00831 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06615E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06650E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06615E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98960E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81107E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81306E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65276E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64919E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10312E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10072E-01 0.00043 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94297E-03 0.00554  2.82885E-04 0.02059  9.22551E-04 0.01085  7.14013E-04 0.01249  1.58239E-03 0.00877  3.59380E-04 0.01755  8.17483E-05 0.03514 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10230E-01 0.01319  1.24805E-02 0.00011  3.21963E-02 0.00016  1.06770E-01 0.00052  3.01467E-01 0.00032  1.25624E+00 0.00100  6.61368E+00 0.02337 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48408E-03 0.00733  2.46770E-04 0.02861  8.11439E-04 0.01602  6.22371E-04 0.01739  1.41686E-03 0.01147  3.16487E-04 0.02632  7.01540E-05 0.04827 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.07862E-01 0.01848  1.24814E-02 0.00014  3.21940E-02 0.00022  1.06781E-01 0.00072  3.01404E-01 0.00045  1.25791E+00 0.00135  7.12023E+00 0.01972 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.37186E-07 0.00132  8.36669E-07 0.00132  9.77960E-07 0.02205 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.58917E-07 0.00124  7.58447E-07 0.00124  8.86745E-07 0.02220 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46691E-03 0.00842  2.46049E-04 0.03356  8.10640E-04 0.01916  6.22857E-04 0.02042  1.40708E-03 0.01412  3.09088E-04 0.03068  7.11977E-05 0.06128 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07903E-01 0.02468  1.24811E-02 0.00022  3.21866E-02 0.00028  1.06669E-01 0.00094  3.01492E-01 0.00055  1.25368E+00 0.00204  7.06425E+00 0.02902 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.33960E-07 0.01966  7.33780E-07 0.01966  7.29456E-07 0.05095 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.65631E-07 0.01965  6.65468E-07 0.01965  6.61784E-07 0.05096 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08546E-03 0.04275  2.20886E-04 0.15666  6.94899E-04 0.08854  5.64394E-04 0.08665  1.26665E-03 0.06210  2.92486E-04 0.12665  4.61466E-05 0.29198 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.16793E-01 0.09373  1.24886E-02 0.00108  3.21891E-02 0.00106  1.07066E-01 0.00363  3.01356E-01 0.00201  1.24525E+00 0.00662  8.37459E+00 0.06924 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08447E-03 0.04224  2.25994E-04 0.15190  6.96374E-04 0.08803  5.64219E-04 0.08696  1.25356E-03 0.06012  2.97071E-04 0.12274  4.72572E-05 0.29222 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.13058E-01 0.08757  1.24888E-02 0.00108  3.21891E-02 0.00106  1.07072E-01 0.00364  3.01425E-01 0.00200  1.24527E+00 0.00647  8.37459E+00 0.06924 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.20169E+03 0.03803 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.29923E-07 0.00082 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.52329E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.50816E-03 0.00524 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.22788E+03 0.00529 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99100E-09 0.00092 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84739E-05 0.00633  6.83706E-05 0.00632  3.38045E-06 0.30199 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.19230E-05 0.02000  6.20176E-05 0.02014  2.29890E-06 0.50247 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.51207E-04 0.01892  2.51100E-04 0.01895  2.62428E-04 0.29754 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63062E+01 0.01206 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45807E+01 0.00027  3.86256E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Thu Apr 28 15:25:49 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 03:54:40 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651170349610 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02045E+00  1.00399E+00  1.00457E+00  1.00264E+00  9.97415E-01  9.97557E-01  1.00197E+00  1.00036E+00  9.98346E-01  1.00088E+00  9.96660E-01  1.00183E+00  9.94886E-01  9.91647E-01  1.00014E+00  1.00078E+00  9.94935E-01  9.90946E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.84092E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61591E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34723E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49096E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67413E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47198E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46262E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.32079E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.32352E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333854 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33388E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33388E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.98247E+03 ;
RUNNING_TIME              (idx, 1)        =  7.48852E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.04650E+00  1.04650E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.11370E+00  1.53133E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.45084E+02  3.14794E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.64200E-01  2.02000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.30471E+01  2.38187E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.46510E+02  7.46510E+02 ];
CPU_USAGE                 (idx, 1)        = 9.32424 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98501E+00 3.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26123E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22100.63;
MEMSIZE                   (idx, 1)        = 21935.46;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 165.18;

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

TOT_ACTIVITY              (idx, 1)        =  6.90695E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73710E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59327E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90625E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32791E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96653E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57033E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65549E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61319E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47103E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16156E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78546E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44000E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03107E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33610E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30980E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51844E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04057E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13403E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21883E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01514E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.38190E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98794E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.84130E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16428E-01 -4.77868E+28  2.68584E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01141E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.49796E+18 0.00435  1.60123E-02 0.00426 ];
U233_FISS                 (idx, [1:   4]) = [  7.92081E+19 0.00057  8.46741E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.07652E+18 0.00191  7.56507E-02 0.00188 ];
U238_FISS                 (idx, [1:   4]) = [  2.08821E+14 0.34941  2.23497E-06 0.34942 ];
PU239_FISS                (idx, [1:   4]) = [  6.55266E+17 0.00616  7.00499E-03 0.00615 ];
PU240_FISS                (idx, [1:   4]) = [  8.73175E+16 0.01756  9.33454E-04 0.01756 ];
PU241_FISS                (idx, [1:   4]) = [  1.53840E+17 0.01304  1.64414E-03 0.01298 ];
TH232_CAPT                (idx, [1:   4]) = [  9.18343E+19 0.00056  6.39401E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15522E+19 0.00133  8.04349E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48784E+18 0.00341  1.73220E-02 0.00339 ];
U238_CAPT                 (idx, [1:   4]) = [  2.58982E+15 0.10149  1.80065E-05 0.10145 ];
PU239_CAPT                (idx, [1:   4]) = [  3.03042E+17 0.00938  2.11013E-03 0.00940 ];
PU240_CAPT                (idx, [1:   4]) = [  2.28217E+17 0.01057  1.58911E-03 0.01058 ];
PU241_CAPT                (idx, [1:   4]) = [  3.24693E+16 0.02670  2.26057E-04 0.02667 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81377E+17 0.01208  1.26296E-03 0.01210 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001643 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08611E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001643 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5489661 5.49497E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3575804 3.57894E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 936178 9.36953E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001643 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.23517E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36683E+20 4.8E-06  2.36683E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35891E+19 9.4E-08  9.35891E+19 9.4E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43575E+20 0.00027  1.27798E+20 0.00019  1.57768E+19 0.00181 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37164E+20 0.00016  2.21387E+20 0.00011  1.57768E+19 0.00181 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61377E+20 0.00035  2.61377E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53184E+22 0.00025  3.34795E+22 0.00023  1.83888E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.44904E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61654E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16764E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.71318E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.71318E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.64288E-01 0.03663 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.04256E-01 0.01046 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.00907E-04 0.01892 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.09328E+03 0.03123 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06311E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.75278E-01 0.02170 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.93304E-01 0.02170 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52896E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 9.4E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05164E-01 0.00043  1.00218E-01 0.00044  3.48897E-04 0.00856 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05550E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05559E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05550E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99168E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81581E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81485E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64496E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64624E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10197E-01 0.00136 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10137E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.97921E-03 0.00553  2.87487E-04 0.01935  9.19147E-04 0.01074  7.15756E-04 0.01285  1.60415E-03 0.00894  3.70823E-04 0.01664  8.18445E-05 0.03854 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13313E-01 0.01503  1.24790E-02 6.9E-05  3.22023E-02 0.00017  1.06785E-01 0.00053  3.01292E-01 0.00029  1.25781E+00 0.00105  6.47599E+00 0.02599 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.50427E-03 0.00818  2.48907E-04 0.02872  8.15290E-04 0.01550  6.37263E-04 0.01868  1.40370E-03 0.01262  3.25563E-04 0.02667  7.35446E-05 0.05601 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.14635E-01 0.02155  1.24783E-02 4.0E-05  3.22022E-02 0.00023  1.06874E-01 0.00084  3.01213E-01 0.00042  1.25680E+00 0.00134  7.07655E+00 0.02109 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.49066E-07 0.00144  8.48547E-07 0.00145  9.96388E-07 0.03419 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.68489E-07 0.00134  7.68021E-07 0.00135  9.01419E-07 0.03385 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46388E-03 0.00867  2.34770E-04 0.03537  7.99900E-04 0.01957  6.39349E-04 0.02029  1.39236E-03 0.01372  3.27553E-04 0.02901  6.99556E-05 0.06611 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.12189E-01 0.02559  1.24800E-02 0.00025  3.21899E-02 0.00028  1.06928E-01 0.00097  3.01298E-01 0.00054  1.25542E+00 0.00204  7.15701E+00 0.02986 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.43861E-07 0.01961  7.43553E-07 0.01961  8.13716E-07 0.04858 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.73268E-07 0.01960  6.72991E-07 0.01960  7.36279E-07 0.04860 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14744E-03 0.04367  2.41760E-04 0.14577  7.62305E-04 0.08205  5.50552E-04 0.09083  1.24177E-03 0.06765  3.11888E-04 0.11443  3.91657E-05 0.26336 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42875E-01 0.06877  1.24742E-02 0.00019  3.21964E-02 0.00098  1.06903E-01 0.00335  3.00111E-01 0.00187  1.25137E+00 0.00435  5.94168E+00 0.11237 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13766E-03 0.04289  2.35895E-04 0.14400  7.58451E-04 0.08043  5.53609E-04 0.09120  1.24541E-03 0.06704  3.05606E-04 0.11094  3.86852E-05 0.24538 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.38722E-01 0.06668  1.24742E-02 0.00019  3.21946E-02 0.00098  1.06883E-01 0.00332  3.00207E-01 0.00187  1.25190E+00 0.00436  5.87620E+00 0.11431 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.24267E+03 0.03920 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.41423E-07 0.00080 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.61579E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.58268E-03 0.00609 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.25839E+03 0.00611 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.11980E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.94950E-05 0.00551  6.94604E-05 0.00553  3.71782E-06 0.27194 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.24120E-05 0.01944  6.23665E-05 0.01938  3.59393E-06 0.32057 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.96610E-04 0.01839  2.96289E-04 0.01834  3.90234E-04 0.25495 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61449E+01 0.01074 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46262E+01 0.00026  3.86509E+01 0.00039 ];

