
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 01:04:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02164E+00  9.96193E-01  9.99784E-01  9.99083E-01  1.00002E+00  1.01050E+00  1.00647E+00  9.91353E-01  9.99700E-01  9.95781E-01  9.99029E-01  9.94042E-01  1.00155E+00  9.97702E-01  9.95233E-01  9.96046E-01  9.98853E-01  9.97016E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.86483E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61352E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31166E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45617E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75485E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48566E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47587E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.47454E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.38923E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333873 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33375E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33375E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.89325E+02 ;
RUNNING_TIME              (idx, 1)        =  2.99975E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.81300E-01  1.81300E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.87406E+01  2.87406E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.20667E-02  4.35333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.99927E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.64498 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98465E+00 3.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.63690E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  2.68477E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.11135E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.05967E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.68477E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.11135E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.58045E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.36978E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.58045E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.36978E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.24234E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.05873E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09484E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.23085E+15 0.00039  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15175E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.77181E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.55822E+18 0.00396  1.66082E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  9.22607E+19 0.00049  9.83392E-01 6.6E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26767E+19 0.00057  7.48323E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33891E+19 0.00138  1.08113E-01 0.00133 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001263 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15320E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001263 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5132853 5.13823E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3888500 3.89249E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 979910 9.80813E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001263 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.86265E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36456E+20 3.8E-06  2.36456E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37769E+19 1.2E-07  9.37769E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23858E+20 0.00029  1.09225E+20 0.00023  1.46329E+19 0.00157 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17635E+20 0.00016  2.03002E+20 0.00012  1.46329E+19 0.00157 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41028E+20 0.00039  2.41028E+20 0.00039  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32396E+22 0.00026  3.14516E+22 0.00024  1.78790E+21 0.00160 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36414E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41276E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07998E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29423E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.13188E-01 0.07547 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.84141E-01 0.00814 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.43439E-04 0.01424 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.15213E+04 0.02185 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01930E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99987E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.04413E-01 0.06224 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.55000E-01 0.06224 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52147E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.81650E-01 0.00041  1.08698E-01 0.00040  3.52740E-04 0.00871 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.81164E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.81073E-01 0.00039 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.81164E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08788E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03068E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03020E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32693E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32732E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48985E-01 0.00163 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49525E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30263E-03 0.00579  2.77227E-04 0.01918  8.22266E-04 0.01103  5.95390E-04 0.01384  1.28738E-03 0.00880  2.67754E-04 0.01815  5.26145E-05 0.04397 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.46392E-01 0.01448  1.24758E-02 2.8E-05  3.23349E-02 9.2E-05  1.06096E-01 0.00058  2.97407E-01 0.00026  1.23586E+00 0.00023  5.28311E+00 0.03841 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24141E-03 0.00848  2.68141E-04 0.02764  8.02340E-04 0.01602  5.80243E-04 0.02138  1.27528E-03 0.01328  2.62247E-04 0.02614  5.31590E-05 0.06719 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.47717E-01 0.02116  1.24759E-02 4.1E-05  3.23379E-02 0.00013  1.06021E-01 0.00086  2.97307E-01 0.00037  1.23575E+00 0.00031  6.33958E+00 0.02971 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.85053E-07 0.00132  8.84722E-07 0.00132  9.82509E-07 0.01676 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.68766E-07 0.00125  8.68442E-07 0.00125  9.64313E-07 0.01673 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23121E-03 0.00883  2.69417E-04 0.03185  7.99145E-04 0.01770  5.81396E-04 0.02060  1.27790E-03 0.01406  2.58555E-04 0.03066  4.47997E-05 0.07536 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.33540E-01 0.02361  1.24752E-02 5.7E-05  3.23350E-02 0.00017  1.06009E-01 0.00096  2.97303E-01 0.00044  1.23631E+00 0.00038  6.43842E+00 0.04340 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.83692E-07 0.01956  7.83416E-07 0.01956  8.73816E-07 0.05812 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.69414E-07 0.01956  7.69144E-07 0.01956  8.57120E-07 0.05799 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89438E-03 0.04320  2.65321E-04 0.13495  7.58001E-04 0.07795  5.34717E-04 0.09266  1.08004E-03 0.06500  2.19348E-04 0.13429  3.69572E-05 0.33819 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.23936E-01 0.09277  1.24764E-02 0.00012  3.23517E-02 0.00059  1.05543E-01 0.00272  2.97442E-01 0.00167  1.23466E+00 0.00119  6.75662E+00 0.14230 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.88958E-03 0.04269  2.67991E-04 0.13449  7.56826E-04 0.07666  5.31972E-04 0.09290  1.07464E-03 0.06365  2.23721E-04 0.12894  3.44260E-05 0.32026 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.25791E-01 0.09551  1.24765E-02 0.00012  3.23491E-02 0.00058  1.05564E-01 0.00274  2.97381E-01 0.00164  1.23468E+00 0.00119  6.75662E+00 0.14230 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.70583E+03 0.03868 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.78478E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.62312E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18855E-03 0.00499 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.63068E+03 0.00514 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.24951E-09 0.00127 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.28757E-05 0.00422  7.28771E-05 0.00421  5.06360E-06 0.21917 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.82458E-05 0.01366  7.83142E-05 0.01371  4.36104E-06 0.28364 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.19324E-04 0.01371  5.18939E-04 0.01379  6.28425E-04 0.21228 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.76592E+01 0.01189 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47587E+01 0.00026  4.03643E+01 0.00040 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 01:49:41 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01104E+00  1.00278E+00  1.00521E+00  1.00095E+00  9.92623E-01  9.99432E-01  9.95219E-01  9.92789E-01  1.00506E+00  1.00216E+00  9.95616E-01  1.00219E+00  1.00760E+00  1.00914E+00  9.92716E-01  9.93367E-01  9.89928E-01  1.00218E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.85672E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61433E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30919E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45348E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75440E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48957E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47975E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.49199E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.38769E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333380 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33366E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33366E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.36236E+02 ;
RUNNING_TIME              (idx, 1)        =  7.48469E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.02383E-01  1.21083E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.34445E+01  4.47039E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.80500E-02  1.80500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.37350E-01  4.52167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.48420E+01  4.50279E+02 ];
CPU_USAGE                 (idx, 1)        = 9.83656 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98887E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.83824E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  5.77795E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60418E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.66147E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.48403E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.04596E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26282E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46839E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.45735E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65677E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.39516E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.79778E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06218E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17699E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44219E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35376E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17373E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55145E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60175E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38114E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39726E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51675E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.52619E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.30967E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.30684E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26594E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.11356E-04 -2.39611E+25  2.15199E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.89433E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.57032E+18 0.00401  1.67357E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  9.22276E+19 0.00051  9.82940E-01 6.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.15703E+14 0.25055  4.43649E-06 0.25030 ];
TH232_CAPT                (idx, [1:   4]) = [  9.38729E+19 0.00054  7.44658E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33860E+19 0.00128  1.06187E-01 0.00121 ];
U235_CAPT                 (idx, [1:   4]) = [  1.22259E+14 0.44425  9.66671E-07 0.44428 ];
SM149_CAPT                (idx, [1:   4]) = [  1.30948E+16 0.04259  1.03902E-04 0.04256 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000976 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15415E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000976 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5170287 5.17578E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3848289 3.85237E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 982400 9.83390E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000976 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.30968E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36450E+20 3.3E-06  2.36450E+20 3.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.2E-07  9.37766E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26105E+20 0.00031  1.11379E+20 0.00023  1.47251E+19 0.00170 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19881E+20 0.00018  2.05156E+20 0.00012  1.47251E+19 0.00170 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43561E+20 0.00035  2.43561E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36204E+22 0.00025  3.18247E+22 0.00023  1.79563E+21 0.00164 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39521E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43833E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09228E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29514E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29514E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.55449E-01 0.07115 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.90625E-01 0.00828 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.62137E-04 0.01569 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.07654E+04 0.02586 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01669E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 9.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.35191E-01 0.05822 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.82544E-01 0.05822 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52141E+00 3.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.71314E-01 0.00043  1.07571E-01 0.00042  3.54358E-04 0.00861 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.70857E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.70836E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.70857E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07675E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03251E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03164E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32442E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32539E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49754E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50227E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34963E-03 0.00538  2.79475E-04 0.01863  8.17577E-04 0.01085  6.16518E-04 0.01267  1.31129E-03 0.00876  2.73044E-04 0.02036  5.17287E-05 0.04427 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43471E-01 0.01523  1.24757E-02 2.7E-05  3.23400E-02 9.1E-05  1.06063E-01 0.00058  2.97319E-01 0.00026  1.23586E+00 0.00021  5.10311E+00 0.04008 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25446E-03 0.00821  2.74538E-04 0.02841  7.93187E-04 0.01729  5.83201E-04 0.01998  1.29117E-03 0.01324  2.61967E-04 0.03052  5.03951E-05 0.06588 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42331E-01 0.02063  1.24760E-02 3.4E-05  3.23365E-02 0.00011  1.06012E-01 0.00078  2.97313E-01 0.00038  1.23580E+00 0.00030  6.45084E+00 0.03019 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.94948E-07 0.00121  8.94667E-07 0.00121  9.79798E-07 0.01565 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.69223E-07 0.00112  8.68951E-07 0.00112  9.51523E-07 0.01562 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28385E-03 0.00885  2.70430E-04 0.03387  8.13190E-04 0.01780  6.11589E-04 0.01977  1.27244E-03 0.01418  2.66078E-04 0.03036  5.01235E-05 0.07597 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.39228E-01 0.02537  1.24758E-02 4.9E-05  3.23357E-02 0.00016  1.05982E-01 0.00087  2.97416E-01 0.00046  1.23604E+00 0.00036  6.19010E+00 0.04326 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.94764E-07 0.01967  7.94396E-07 0.01967  8.60809E-07 0.05122 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.72107E-07 0.01967  7.71749E-07 0.01967  8.36235E-07 0.05123 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00281E-03 0.04575  2.46661E-04 0.13462  8.31116E-04 0.07886  5.52418E-04 0.09425  1.04577E-03 0.06758  2.88589E-04 0.13490  3.82597E-05 0.32689 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.14278E-01 0.09037  1.24763E-02 0.00013  3.23223E-02 0.00041  1.06427E-01 0.00359  2.97221E-01 0.00164  1.23565E+00 0.00111  6.26138E+00 0.15198 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00343E-03 0.04519  2.51152E-04 0.13089  8.27815E-04 0.07842  5.39592E-04 0.09145  1.05147E-03 0.06626  2.93623E-04 0.13214  3.97795E-05 0.32307 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.14955E-01 0.08896  1.24763E-02 0.00013  3.23219E-02 0.00040  1.06445E-01 0.00363  2.97222E-01 0.00164  1.23581E+00 0.00109  6.26138E+00 0.15198 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.78193E+03 0.04115 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.89973E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.64398E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.27353E-03 0.00566 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.67923E+03 0.00578 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16770E-09 0.00123 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.17224E-05 0.00408  7.17069E-05 0.00410  2.15965E-06 0.36216 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.78717E-05 0.01753  7.79482E-05 0.01751  8.05042E-07 0.49732 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.40614E-04 0.01516  4.41291E-04 0.01521  2.42769E-04 0.35232 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71326E+01 0.01079 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47975E+01 0.00026  4.04109E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 02:37:15 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00966E+00  9.99760E-01  1.00719E+00  9.98599E-01  9.95739E-01  9.94853E-01  1.00444E+00  9.95220E-01  1.00162E+00  9.98653E-01  9.99358E-01  1.00051E+00  9.99809E-01  1.00392E+00  9.97997E-01  9.98991E-01  9.95220E-01  9.98457E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.83362E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61664E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31095E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45448E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75200E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50682E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49706E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.52105E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.37804E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334255 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33440E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33440E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.21027E+03 ;
RUNNING_TIME              (idx, 1)        =  1.22418E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.35383E-01  1.33000E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.20856E+02  4.74114E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.61333E-02  1.80833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.87067E-01  4.96500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.22413E+02  6.60673E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88640 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98915E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89053E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.40056E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65530E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.26837E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.87318E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31024E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49595E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49276E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.77293E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.40408E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.87702E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.04072E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89568E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60000E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49272E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34369E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33192E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54076E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.70171E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17638E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.01825E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51678E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.36846E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.53865E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.39771E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61906E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.92647E-04 -1.27523E+26  2.15302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.08348E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.58928E+18 0.00398  1.69619E-02 0.00388 ];
U233_FISS                 (idx, [1:   4]) = [  9.19626E+19 0.00050  9.81577E-01 6.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.20801E+16 0.04248  1.28924E-04 0.04248 ];
TH232_CAPT                (idx, [1:   4]) = [  9.53653E+19 0.00050  7.38675E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33857E+19 0.00133  1.03683E-01 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  4.17735E+15 0.07955  3.23082E-05 0.07955 ];
SM149_CAPT                (idx, [1:   4]) = [  6.28580E+16 0.01970  4.86937E-04 0.01970 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003188 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16558E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003188 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5230990 5.23557E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3796218 3.79940E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 975980 9.76679E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003188 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.11759E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36433E+20 3.5E-06  2.36433E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37760E+19 1.2E-07  9.37760E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29118E+20 0.00031  1.14121E+20 0.00023  1.49972E+19 0.00167 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22894E+20 0.00018  2.07897E+20 0.00013  1.49972E+19 0.00167 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46590E+20 0.00036  2.46590E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.41071E+22 0.00027  3.22833E+22 0.00024  1.82379E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40848E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46979E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11017E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29739E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29739E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.92123E-01 0.05176 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.03186E-01 0.01094 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.94028E-04 0.01937 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.14160E+03 0.02762 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02336E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.06826E-01 0.03250 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.27995E-01 0.03250 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52126E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.57806E-01 0.00045  1.06093E-01 0.00043  3.42039E-04 0.00906 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.58415E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.58848E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.58415E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06210E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04297E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04203E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31067E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31170E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50313E-01 0.00167 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50503E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38882E-03 0.00576  2.75995E-04 0.01925  8.41625E-04 0.01119  6.13063E-04 0.01337  1.33088E-03 0.00895  2.77976E-04 0.01979  4.92797E-05 0.04661 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42901E-01 0.01546  1.24755E-02 2.7E-05  3.23370E-02 9.2E-05  1.06136E-01 0.00058  2.97367E-01 0.00025  1.23583E+00 0.00023  5.11554E+00 0.04188 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25489E-03 0.00888  2.70095E-04 0.02889  7.97611E-04 0.01609  5.96232E-04 0.01954  1.27352E-03 0.01384  2.66352E-04 0.02889  5.10831E-05 0.07502 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.47647E-01 0.02285  1.24757E-02 3.6E-05  3.23416E-02 0.00015  1.05938E-01 0.00073  2.97266E-01 0.00035  1.23547E+00 0.00031  6.51826E+00 0.02992 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.13885E-07 0.00132  9.13509E-07 0.00131  1.02869E-06 0.01914 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.75262E-07 0.00121  8.74902E-07 0.00121  9.85053E-07 0.01910 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21112E-03 0.00940  2.69142E-04 0.03323  8.01584E-04 0.01986  5.72432E-04 0.02229  1.26465E-03 0.01457  2.55676E-04 0.03306  4.76343E-05 0.07546 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.48687E-01 0.02627  1.24745E-02 6.2E-05  3.23367E-02 0.00016  1.06187E-01 0.00120  2.97389E-01 0.00042  1.23533E+00 0.00039  7.07044E+00 0.03861 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.11142E-07 0.01970  8.10852E-07 0.01970  9.22951E-07 0.05031 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.77191E-07 0.01970  7.76914E-07 0.01970  8.84275E-07 0.05030 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89360E-03 0.04454  2.18055E-04 0.13780  6.97128E-04 0.07568  5.26515E-04 0.09368  1.20417E-03 0.06873  2.18844E-04 0.13702  2.88931E-05 0.33340 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42359E-01 0.10081  1.24755E-02 0.00014  3.23650E-02 0.00063  1.05742E-01 0.00285  2.97191E-01 0.00154  1.23675E+00 0.00106  8.62325E+00 0.09779 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.88473E-03 0.04355  2.29768E-04 0.13260  6.83636E-04 0.07348  5.32649E-04 0.09219  1.19215E-03 0.06707  2.15505E-04 0.13535  3.10281E-05 0.33616 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.44333E-01 0.10095  1.24755E-02 0.00014  3.23663E-02 0.00063  1.05746E-01 0.00283  2.97152E-01 0.00153  1.23682E+00 0.00106  8.62325E+00 0.09779 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.59691E+03 0.04094 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.07834E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.69468E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21316E-03 0.00641 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.54011E+03 0.00649 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.98639E-09 0.00099 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92409E-05 0.00544  6.92124E-05 0.00544  1.89640E-06 0.38461 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.70076E-05 0.02001  6.69359E-05 0.02010  1.88457E-06 0.54612 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.80517E-04 0.01941  2.80768E-04 0.01944  2.15154E-04 0.37620 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74543E+01 0.01199 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49706E+01 0.00026  4.06191E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 03:25:30 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01674E+00  9.99431E-01  1.00143E+00  9.99916E-01  9.98681E-01  1.01021E+00  1.00095E+00  9.89380E-01  9.93604E-01  9.94084E-01  9.96917E-01  9.95437E-01  1.00948E+00  9.96319E-01  9.98941E-01  9.96789E-01  9.95687E-01  1.00601E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.81689E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61831E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31265E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45560E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.74140E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51527E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50550E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.53281E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.36986E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333455 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33384E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33384E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.69101E+03 ;
RUNNING_TIME              (idx, 1)        =  1.70659E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.66600E-01  1.31217E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.68937E+02  4.80811E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.40333E-02  1.79000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.27167E-02  1.27167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.32567E-01  4.54333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.70653E+02  6.93819E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90870 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98904E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91396E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.51992E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67139E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.27970E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91221E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33658E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57600E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50596E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.83388E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.56682E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.03834E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.24119E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79441E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74268E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48797E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36111E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34525E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55870E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39745E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42545E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09298E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54590E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.30139E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70352E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.47039E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.15880E-03 -2.49345E+26  2.15424E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.21142E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.61606E+18 0.00409  1.72236E-02 0.00403 ];
U233_FISS                 (idx, [1:   4]) = [  9.19302E+19 0.00051  9.79799E-01 7.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.68151E+16 0.02347  4.98950E-04 0.02341 ];
TH232_CAPT                (idx, [1:   4]) = [  9.63910E+19 0.00055  7.35144E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33939E+19 0.00132  1.02154E-01 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  1.70477E+16 0.03637  1.30000E-04 0.03637 ];
SM149_CAPT                (idx, [1:   4]) = [  1.03384E+17 0.01565  7.88637E-04 0.01569 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001508 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13744E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001508 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5260110 5.26554E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3764359 3.76794E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 977039 9.77891E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001508 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.45058E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36430E+20 3.7E-06  2.36430E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.2E-07  9.37752E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31147E+20 0.00031  1.15929E+20 0.00023  1.52186E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.24923E+20 0.00018  2.09704E+20 0.00012  1.52186E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.49013E+20 0.00037  2.49013E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44543E+22 0.00027  3.26072E+22 0.00024  1.84702E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43516E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.49274E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12314E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29707E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29707E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.99868E-01 0.04879 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24786E-01 0.01092 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.25706E-04 0.02203 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.85510E+03 0.02787 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02214E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.86330E-01 0.03369 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.09382E-01 0.03369 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52124E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.50002E-01 0.00045  1.05213E-01 0.00043  3.41647E-04 0.00876 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.49584E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.49507E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.49584E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05252E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04300E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04438E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31063E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30863E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52721E-01 0.00165 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52052E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.43157E-03 0.00568  2.86431E-04 0.01977  8.40395E-04 0.01110  6.33639E-04 0.01316  1.33236E-03 0.00865  2.88310E-04 0.01996  5.04320E-05 0.04430 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.37465E-01 0.01417  1.24754E-02 2.7E-05  3.23349E-02 9.1E-05  1.06231E-01 0.00062  2.97429E-01 0.00026  1.23554E+00 0.00027  4.96851E+00 0.03976 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25717E-03 0.00837  2.77601E-04 0.02905  8.08473E-04 0.01602  6.01817E-04 0.01952  1.24801E-03 0.01311  2.71443E-04 0.02907  4.98295E-05 0.07162 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.41901E-01 0.02312  1.24750E-02 4.1E-05  3.23346E-02 0.00014  1.06127E-01 0.00081  2.97368E-01 0.00037  1.23620E+00 0.00043  6.03918E+00 0.03110 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.23452E-07 0.00121  9.23109E-07 0.00121  1.03244E-06 0.01571 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.77229E-07 0.00113  8.76903E-07 0.00113  9.80697E-07 0.01569 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23943E-03 0.00877  2.73597E-04 0.02970  7.89853E-04 0.01814  5.95539E-04 0.01979  1.25882E-03 0.01490  2.73579E-04 0.03240  4.80439E-05 0.07313 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.36452E-01 0.02333  1.24752E-02 5.6E-05  3.23360E-02 0.00016  1.06123E-01 0.00094  2.97543E-01 0.00048  1.23704E+00 0.00059  5.88251E+00 0.04430 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.15096E-07 0.01971  8.15121E-07 0.01971  7.79975E-07 0.05162 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.74490E-07 0.01971  7.74514E-07 0.01971  7.40935E-07 0.05160 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93541E-03 0.04581  2.70022E-04 0.12264  7.70143E-04 0.08270  4.69162E-04 0.09619  1.12221E-03 0.06613  2.51167E-04 0.13650  5.26975E-05 0.26153 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.55305E-01 0.11899  1.24751E-02 0.00014  3.23424E-02 0.00055  1.06418E-01 0.00366  2.97331E-01 0.00162  1.23559E+00 0.00112  6.93907E+00 0.11759 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93243E-03 0.04511  2.62707E-04 0.12053  7.66626E-04 0.08126  4.67796E-04 0.09450  1.13629E-03 0.06631  2.41703E-04 0.13202  5.73077E-05 0.25529 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.64740E-01 0.12125  1.24751E-02 0.00014  3.23437E-02 0.00056  1.06438E-01 0.00369  2.97348E-01 0.00162  1.23573E+00 0.00111  6.93907E+00 0.11759 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.62532E+03 0.04232 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.14486E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.68706E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25369E-03 0.00622 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.55856E+03 0.00627 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91453E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.80869E-05 0.00611  6.80801E-05 0.00614  2.21834E-06 0.31668 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.16463E-05 0.02246  6.17476E-05 0.02250  1.86882E-06 0.41429 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.16916E-04 0.02122  2.16658E-04 0.02136  2.92159E-04 0.31180 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70688E+01 0.01110 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50550E+01 0.00027  4.07169E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 04:15:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00558E+00  1.00018E+00  9.91582E-01  1.00092E+00  1.00139E+00  9.98853E-01  1.00215E+00  1.00544E+00  1.00103E+00  1.00124E+00  9.96085E-01  9.99980E-01  1.00503E+00  9.99093E-01  9.93939E-01  9.95766E-01  9.99573E-01  1.00217E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.80678E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61932E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31258E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45522E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73967E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53367E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52392E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.56911E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.37121E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334164 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33381E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33381E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.18325E+03 ;
RUNNING_TIME              (idx, 1)        =  2.20205E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.03333E-01  1.36733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.18313E+02  4.93761E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.16500E-02  1.76167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.27167E-02  1.27167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.32283E-01  1.99667E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.20047E+02  7.02402E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91460 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98854E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92032E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.54731E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66619E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.67183E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93508E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35086E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58016E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49898E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00265E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.68746E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.32239E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.34442E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69815E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85290E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93614E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32449E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31718E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52005E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34906E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81408E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09069E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52036E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.56901E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.75082E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.57380E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64039E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.35168E-03 -5.06022E+26  2.15681E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.45489E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.62190E+18 0.00394  1.72783E-02 0.00388 ];
U233_FISS                 (idx, [1:   4]) = [  9.16284E+19 0.00055  9.76143E-01 7.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.76791E+17 0.01152  1.88328E-03 0.01149 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81355E+19 0.00051  7.31045E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33443E+19 0.00136  9.94062E-02 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  6.33846E+16 0.02063  4.72100E-04 0.02060 ];
SM149_CAPT                (idx, [1:   4]) = [  1.42315E+17 0.01427  1.06009E-03 0.01426 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001420 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15360E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001420 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5311828 5.31731E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3714398 3.71814E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 975194 9.76079E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001420 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.52737E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36415E+20 3.8E-06  2.36415E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34273E+20 0.00031  1.18801E+20 0.00023  1.54718E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.28046E+20 0.00018  2.12574E+20 0.00013  1.54718E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.52460E+20 0.00035  2.52460E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50235E+22 0.00025  3.31475E+22 0.00022  1.87596E+21 0.00171 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46429E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.52689E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14336E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29343E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29343E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.44324E-01 0.04859 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.16520E-01 0.01307 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.02331E-04 0.02280 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.14587E+03 0.02688 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02396E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.82365E-01 0.03310 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.05980E-01 0.03310 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52115E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.37119E-01 0.00047  1.03817E-01 0.00045  3.38179E-04 0.00948 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.36695E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.36481E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.36695E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03804E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04596E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04485E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30673E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30801E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54333E-01 0.00171 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54347E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.50304E-03 0.00593  2.81606E-04 0.01938  8.72805E-04 0.01126  6.35436E-04 0.01279  1.37339E-03 0.00932  2.83732E-04 0.01957  5.60730E-05 0.04574 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.48548E-01 0.01575  1.24760E-02 2.8E-05  3.23296E-02 8.6E-05  1.06216E-01 0.00063  2.97711E-01 0.00028  1.23258E+00 0.00336  5.18548E+00 0.03917 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26055E-03 0.00871  2.67195E-04 0.02910  8.00739E-04 0.01638  6.06425E-04 0.01890  1.28061E-03 0.01350  2.54810E-04 0.02844  5.07721E-05 0.07553 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.36431E-01 0.02361  1.24766E-02 2.8E-05  3.23277E-02 0.00012  1.06109E-01 0.00091  2.97557E-01 0.00037  1.23686E+00 0.00032  6.39429E+00 0.02949 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.38950E-07 0.00118  9.38601E-07 0.00119  1.04717E-06 0.01808 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.79849E-07 0.00109  8.79522E-07 0.00109  9.81360E-07 0.01810 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24446E-03 0.00963  2.55946E-04 0.02949  8.09715E-04 0.01957  5.96349E-04 0.02110  1.27334E-03 0.01575  2.57222E-04 0.03386  5.18875E-05 0.07684 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44011E-01 0.02510  1.24762E-02 4.2E-05  3.23258E-02 0.00015  1.06294E-01 0.00106  2.97661E-01 0.00048  1.23679E+00 0.00049  6.44072E+00 0.04139 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.33962E-07 0.01970  8.33579E-07 0.01970  9.35862E-07 0.06369 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.81742E-07 0.01969  7.81382E-07 0.01969  8.77887E-07 0.06372 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.70428E-03 0.04551  2.15085E-04 0.14635  6.04871E-04 0.08740  5.28953E-04 0.08732  1.00571E-03 0.07149  2.93180E-04 0.13217  5.64797E-05 0.29971 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.73454E-01 0.08972  1.24766E-02 0.00012  3.23135E-02 0.00055  1.05856E-01 0.00298  2.97804E-01 0.00175  1.23614E+00 0.00151  7.51023E+00 0.09606 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.70520E-03 0.04416  2.11196E-04 0.14590  6.08928E-04 0.08301  5.32320E-04 0.08823  9.96989E-04 0.07088  3.00851E-04 0.12817  5.49168E-05 0.29909 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.74485E-01 0.09295  1.24765E-02 0.00012  3.23147E-02 0.00055  1.05879E-01 0.00303  2.97887E-01 0.00175  1.23623E+00 0.00149  7.51023E+00 0.09606 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.25172E+03 0.04136 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.33616E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.74858E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18917E-03 0.00572 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.41674E+03 0.00580 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90089E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.82853E-05 0.00688  6.83019E-05 0.00688  4.05568E-07 0.71660 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.94704E-05 0.02447  5.93495E-05 0.02440  1.17465E-06 0.71790 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.92682E-04 0.02202  1.93157E-04 0.02207  5.85071E-05 0.70600 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72340E+01 0.01267 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52392E+01 0.00028  4.08289E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 05:04:35 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00863E+00  1.00308E+00  9.94347E-01  1.00590E+00  9.97772E-01  9.98810E-01  1.00122E+00  1.00529E+00  9.99516E-01  9.95777E-01  1.00313E+00  9.85562E-01  1.00469E+00  1.00225E+00  9.95425E-01  1.00038E+00  9.95013E-01  1.00321E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.80244E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61976E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31080E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45334E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73946E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54514E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53536E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59794E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.37547E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333551 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33369E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33369E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.67327E+03 ;
RUNNING_TIME              (idx, 1)        =  2.69752E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.44750E-01  1.41417E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.67683E+02  4.93702E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.03000E-02  1.86500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.27167E-02  1.27167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.83733E-01  4.51417E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.69341E+02  7.15167E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91009 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98730E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91496E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.61818E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67626E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.20209E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97198E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37527E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61369E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50619E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10196E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79908E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.74974E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.51356E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25445E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94742E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35112E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33535E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32513E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53086E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02005E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17228E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13211E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55474E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.47269E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85648E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.66041E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96058E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.52650E-03 -7.58813E+26  2.15933E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.67094E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.63215E+18 0.00411  1.73988E-02 0.00408 ];
U233_FISS                 (idx, [1:   4]) = [  9.11544E+19 0.00051  9.71712E-01 9.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.75676E+17 0.00812  4.00535E-03 0.00817 ];
TH232_CAPT                (idx, [1:   4]) = [  9.95449E+19 0.00053  7.27477E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32978E+19 0.00132  9.71814E-02 0.00126 ];
U235_CAPT                 (idx, [1:   4]) = [  1.34647E+17 0.01368  9.84119E-04 0.01369 ];
SM149_CAPT                (idx, [1:   4]) = [  1.61136E+17 0.01183  1.17775E-03 0.01186 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001081 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14886E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001081 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5353098 5.35884E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3670029 3.67378E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 977954 9.78871E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001081 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.45286E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36399E+20 3.7E-06  2.36399E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37698E+19 1.2E-07  9.37698E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36787E+20 0.00030  1.21105E+20 0.00020  1.56821E+19 0.00187 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30557E+20 0.00018  2.14875E+20 0.00011  1.56821E+19 0.00187 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.55347E+20 0.00037  2.55347E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.55008E+22 0.00026  3.36031E+22 0.00023  1.89766E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.49961E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.55553E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15935E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28839E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28839E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.91563E-01 0.04764 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30568E-01 0.01208 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.89122E-04 0.02210 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.94888E+03 0.02779 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02116E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.73466E-01 0.03310 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.97749E-01 0.03310 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52105E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.26275E-01 0.00043  1.02580E-01 0.00043  3.30522E-04 0.00934 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.26118E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.25830E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.26118E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02665E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04440E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04210E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30880E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31160E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56815E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.56812E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.53811E-03 0.00589  2.89085E-04 0.01980  8.72143E-04 0.01151  6.50444E-04 0.01303  1.37409E-03 0.00845  2.96934E-04 0.01869  5.54114E-05 0.04233 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45600E-01 0.01403  1.24754E-02 3.0E-05  3.23255E-02 0.00010  1.06313E-01 0.00062  2.97750E-01 0.00027  1.23691E+00 0.00032  5.15373E+00 0.03751 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19633E-03 0.00844  2.54218E-04 0.02922  7.90581E-04 0.01694  5.86235E-04 0.01938  1.25122E-03 0.01301  2.66386E-04 0.02831  4.76937E-05 0.06718 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38529E-01 0.01997  1.24759E-02 3.4E-05  3.23277E-02 0.00014  1.06199E-01 0.00082  2.97812E-01 0.00041  1.23694E+00 0.00044  6.10803E+00 0.02988 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.50386E-07 0.00117  9.49913E-07 0.00117  1.08896E-06 0.02360 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.80277E-07 0.00112  8.79840E-07 0.00112  1.00858E-06 0.02361 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21172E-03 0.00946  2.61236E-04 0.03283  7.91438E-04 0.01858  5.86599E-04 0.02170  1.24494E-03 0.01509  2.74063E-04 0.03219  5.34399E-05 0.07147 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53775E-01 0.02492  1.24758E-02 4.9E-05  3.23200E-02 0.00016  1.06250E-01 0.00105  2.97705E-01 0.00049  1.23673E+00 0.00059  6.21901E+00 0.04145 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.43507E-07 0.01974  8.43152E-07 0.01974  9.05390E-07 0.06200 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.81304E-07 0.01973  7.80974E-07 0.01973  8.38620E-07 0.06190 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94331E-03 0.04357  1.98701E-04 0.15147  7.05278E-04 0.07993  5.87520E-04 0.09925  1.17994E-03 0.07026  2.34231E-04 0.13541  3.76480E-05 0.28819 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.57369E-01 0.11153  1.24764E-02 0.00014  3.22859E-02 0.00047  1.06503E-01 0.00356  2.98239E-01 0.00185  1.23553E+00 0.00132  5.89810E+00 0.13519 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.95991E-03 0.04360  1.99558E-04 0.15119  6.95608E-04 0.07864  5.98381E-04 0.10090  1.18011E-03 0.06987  2.44766E-04 0.13184  4.14951E-05 0.28951 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.59749E-01 0.10866  1.24764E-02 0.00014  3.22866E-02 0.00046  1.06493E-01 0.00354  2.98281E-01 0.00185  1.23554E+00 0.00132  5.89810E+00 0.13519 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.50891E+03 0.03907 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.45479E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.75716E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21104E-03 0.00652 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.39698E+03 0.00659 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89964E-09 0.00070 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72557E-05 0.00723  6.72597E-05 0.00726  2.04465E-06 0.33641 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.72877E-05 0.02584  5.73454E-05 0.02584  2.06344E-06 0.52794 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.82421E-04 0.02190  1.82170E-04 0.02194  2.76313E-04 0.33083 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73758E+01 0.01092 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53536E+01 0.00027  4.08759E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 05:54:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00271E+00  1.00217E+00  1.00186E+00  1.00209E+00  9.96283E-01  1.00279E+00  9.95410E-01  1.00647E+00  1.00400E+00  9.93132E-01  1.00729E+00  9.97802E-01  9.98527E-01  9.99365E-01  9.93514E-01  1.00301E+00  9.98943E-01  9.94636E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.80250E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61975E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30603E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44874E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73945E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55461E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54478E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.63333E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.38234E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334310 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33417E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33417E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.17163E+03 ;
RUNNING_TIME              (idx, 1)        =  3.19771E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.86317E-01  1.41567E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.17522E+02  4.98385E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.08750E-01  1.84500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.27167E-02  1.27167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.30017E-01  4.62333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.19766E+02  7.16008E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91844 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98925E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92394E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.64955E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67519E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.26244E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00042E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39230E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61625E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50308E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27764E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.93456E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.78903E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.67284E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.94001E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06634E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06898E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30946E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30425E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50273E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.83716E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78547E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13984E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.56859E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.24264E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.90671E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77087E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60097E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.96213E-03 -1.28290E+27  2.16457E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.96235E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.64873E+18 0.00413  1.75845E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  9.01787E+19 0.00051  9.61878E-01 0.00011 ];
U235_FISS                 (idx, [1:   4]) = [  8.98434E+17 0.00527  9.58273E-03 0.00523 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01056E+20 0.00052  7.21439E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31592E+19 0.00138  9.39457E-02 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  3.19389E+17 0.00874  2.28004E-03 0.00872 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77080E+17 0.01152  1.26424E-03 0.01152 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002504 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15127E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002504 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5402754 5.40774E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3616296 3.61943E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 983454 9.84343E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002504 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.80677E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36363E+20 4.0E-06  2.36363E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37618E+19 1.2E-07  9.37618E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40101E+20 0.00030  1.24193E+20 0.00021  1.59081E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33863E+20 0.00018  2.17955E+20 0.00012  1.59081E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59029E+20 0.00035  2.59029E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.60934E+22 0.00024  3.41735E+22 0.00022  1.91987E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.54984E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59361E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17849E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27814E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27814E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.29295E-01 0.04509 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.16422E-01 0.01289 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.83519E-04 0.02500 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.70203E+03 0.02816 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01569E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.62349E-01 0.03310 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.87325E-01 0.03310 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52089E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99703E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.12229E-01 0.00047  1.01048E-01 0.00044  3.28393E-04 0.00932 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12400E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.12531E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12400E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01201E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03851E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03742E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31650E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31776E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.60883E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.61613E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.60907E-03 0.00528  2.89501E-04 0.01852  8.85468E-04 0.01150  6.44418E-04 0.01318  1.42351E-03 0.00874  3.07312E-04 0.01884  5.88596E-05 0.04386 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53281E-01 0.01491  1.24748E-02 3.0E-05  3.23204E-02 0.00010  1.06257E-01 0.00059  2.98116E-01 0.00030  1.23994E+00 0.00042  5.12162E+00 0.03765 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.30352E-03 0.00829  2.60647E-04 0.02799  8.11878E-04 0.01807  5.96809E-04 0.01918  1.30234E-03 0.01339  2.81827E-04 0.02875  5.00240E-05 0.07179 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45849E-01 0.02098  1.24748E-02 4.4E-05  3.23211E-02 0.00016  1.06229E-01 0.00083  2.98204E-01 0.00042  1.24052E+00 0.00061  6.25024E+00 0.02917 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.66858E-07 0.00116  9.66558E-07 0.00117  1.06402E-06 0.01838 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.81949E-07 0.00110  8.81676E-07 0.00111  9.70544E-07 0.01836 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24062E-03 0.00959  2.56991E-04 0.03494  7.89657E-04 0.01990  5.78421E-04 0.02086  1.28051E-03 0.01524  2.79651E-04 0.02893  5.53943E-05 0.06931 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.54699E-01 0.02414  1.24750E-02 5.5E-05  3.23118E-02 0.00018  1.06301E-01 0.00110  2.98143E-01 0.00050  1.23958E+00 0.00066  5.79907E+00 0.04328 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.53945E-07 0.01964  8.53707E-07 0.01964  8.78186E-07 0.05119 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.79041E-07 0.01963  7.78824E-07 0.01963  8.01322E-07 0.05118 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.84499E-03 0.04346  2.04100E-04 0.13772  7.14476E-04 0.08818  4.73203E-04 0.09145  1.10449E-03 0.06336  2.91740E-04 0.12356  5.69826E-05 0.28456 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.37190E-01 0.12784  1.24776E-02 0.00010  3.22956E-02 0.00054  1.05655E-01 0.00270  2.98301E-01 0.00181  1.23880E+00 0.00171  6.25127E+00 0.12124 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85210E-03 0.04285  1.99672E-04 0.13545  7.21725E-04 0.08580  4.74800E-04 0.09065  1.11147E-03 0.06199  2.88450E-04 0.11998  5.59774E-05 0.29284 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.26812E-01 0.12291  1.24776E-02 0.00011  3.22967E-02 0.00054  1.05663E-01 0.00269  2.98308E-01 0.00179  1.23923E+00 0.00175  6.25127E+00 0.12124 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.35026E+03 0.03916 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.59992E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.75677E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20970E-03 0.00569 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.34377E+03 0.00569 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91480E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69479E-05 0.00887  6.69287E-05 0.00887  5.76878E-07 0.70654 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.61435E-05 0.02789  5.62192E-05 0.02797  3.78877E-08 0.72204 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.76254E-04 0.02371  1.76688E-04 0.02369  5.36554E-05 0.70601 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72114E+01 0.01126 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54478E+01 0.00028  4.08795E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 06:45:24 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00599E+00  9.97530E-01  9.96913E-01  9.98956E-01  9.95091E-01  1.00457E+00  9.99720E-01  1.00257E+00  9.98456E-01  9.95483E-01  1.00283E+00  9.98055E-01  1.00171E+00  1.00313E+00  9.99323E-01  9.94836E-01  1.00392E+00  1.00092E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.80797E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61920E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30235E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44528E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.74029E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55839E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54858E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.65382E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.39110E+00 0.00043  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334103 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33395E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33395E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.67764E+03 ;
RUNNING_TIME              (idx, 1)        =  3.70561E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.13418E+00  1.47867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.68123E+02  5.06011E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.27633E-01  1.88833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.66333E-02  1.39167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.76617E-01  4.65500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.70556E+02  7.20593E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92453 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98942E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93058E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.73352E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69059E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.22194E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02800E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40584E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67191E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51644E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.77612E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.23868E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.35514E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.92246E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.24003E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.34317E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32971E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.31096E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30180E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50129E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.56456E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27991E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.18103E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.68546E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.91785E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99566E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.89569E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.20885E-02 -2.60115E+27  2.17776E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03075E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.65832E+18 0.00390  1.76903E-02 0.00383 ];
U233_FISS                 (idx, [1:   4]) = [  8.77881E+19 0.00055  9.36528E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.48699E+18 0.00337  2.65301E-02 0.00328 ];
PU239_FISS                (idx, [1:   4]) = [  1.21449E+15 0.15246  1.29301E-05 0.15241 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01848E+20 0.00055  7.07836E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.27835E+19 0.00137  8.88467E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  8.85612E+17 0.00553  6.15485E-03 0.00551 ];
U238_CAPT                 (idx, [1:   4]) = [  2.66062E+13 1.00000  1.83797E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  3.70780E+14 0.26139  2.57269E-06 0.26140 ];
PU240_CAPT                (idx, [1:   4]) = [  5.21819E+13 1.00000  3.65217E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.86348E+17 0.01184  1.29525E-03 0.01186 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001836 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12952E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001836 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5461754 5.46705E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3558323 3.56162E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981759 9.82624E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001836 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.67638E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36261E+20 4.2E-06  2.36261E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37385E+19 1.2E-07  9.37385E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43896E+20 0.00029  1.27703E+20 0.00021  1.61932E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37634E+20 0.00018  2.21441E+20 0.00012  1.61932E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.63190E+20 0.00035  2.63190E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67255E+22 0.00025  3.47752E+22 0.00023  1.95029E+21 0.00169 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.58620E+19 0.00131 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.63496E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19844E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.25192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.25192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.53517E-01 0.05384 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.37216E-01 0.01283 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93608E-04 0.02493 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.75005E+03 0.02838 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01741E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.96670E-01 0.03786 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.28158E-01 0.03786 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52043E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99753E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.97703E-01 0.00043  9.94101E-02 0.00043  3.30282E-04 0.00940 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.97673E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.97718E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.97673E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95489E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00760E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00579E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35785E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36012E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.71477E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.71638E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.76245E-03 0.00520  2.99827E-04 0.01829  9.15161E-04 0.01077  6.69061E-04 0.01266  1.48632E-03 0.00754  3.22431E-04 0.01876  6.96436E-05 0.03863 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.76979E-01 0.01469  1.24754E-02 2.6E-05  3.22957E-02 0.00012  1.06353E-01 0.00062  2.98963E-01 0.00026  1.24527E+00 0.00056  5.99710E+00 0.02966 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.32752E-03 0.00814  2.62422E-04 0.02706  8.20757E-04 0.01701  5.95047E-04 0.01959  1.30710E-03 0.01234  2.80023E-04 0.02941  6.21750E-05 0.05719 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.77165E-01 0.02147  1.24757E-02 3.6E-05  3.22897E-02 0.00015  1.06472E-01 0.00089  2.98944E-01 0.00041  1.24490E+00 0.00073  6.70176E+00 0.02405 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.73441E-07 0.00130  9.73054E-07 0.00130  1.09304E-06 0.01922 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.73806E-07 0.00121  8.73459E-07 0.00121  9.81116E-07 0.01918 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.31534E-03 0.00953  2.51629E-04 0.03447  8.08266E-04 0.01861  5.99082E-04 0.02235  1.30641E-03 0.01489  2.85765E-04 0.03148  6.41856E-05 0.06374 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.83034E-01 0.02519  1.24762E-02 5.1E-05  3.22984E-02 0.00020  1.06166E-01 0.00096  2.98983E-01 0.00048  1.24748E+00 0.00110  6.64939E+00 0.03536 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.56945E-07 0.01968  8.56470E-07 0.01967  1.02857E-06 0.06713 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.69230E-07 0.01966  7.68802E-07 0.01966  9.23905E-07 0.06722 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91985E-03 0.04440  2.10967E-04 0.13598  7.37429E-04 0.08473  5.27792E-04 0.09430  1.11792E-03 0.06906  2.37519E-04 0.14403  8.82226E-05 0.21728 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.84116E-01 0.10659  1.24759E-02 0.00014  3.23178E-02 0.00063  1.06566E-01 0.00372  2.97811E-01 0.00159  1.24385E+00 0.00272  7.27466E+00 0.07501 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92499E-03 0.04383  2.18405E-04 0.13076  7.41896E-04 0.08387  5.16667E-04 0.09481  1.11430E-03 0.06803  2.46303E-04 0.14278  8.74200E-05 0.21189 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.87503E-01 0.10425  1.24759E-02 0.00014  3.23156E-02 0.00063  1.06572E-01 0.00372  2.97883E-01 0.00160  1.24398E+00 0.00274  7.27466E+00 0.07501 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.42128E+03 0.04017 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.65947E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.67091E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29091E-03 0.00542 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.40791E+03 0.00555 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93860E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70766E-05 0.00709  6.70285E-05 0.00711  1.37219E-06 0.46307 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.96370E-05 0.02366  5.96661E-05 0.02377  9.62548E-07 0.52826 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85809E-04 0.02420  1.86006E-04 0.02421  1.46768E-04 0.44549 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71571E+01 0.01131 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54858E+01 0.00025  4.06446E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 07:36:39 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00970E+00  9.93893E-01  1.00102E+00  1.00110E+00  9.94574E-01  1.00493E+00  1.00066E+00  9.99773E-01  1.00328E+00  9.99670E-01  1.00343E+00  9.92570E-01  1.00010E+00  1.00546E+00  9.98430E-01  1.00003E+00  9.90625E-01  1.00075E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.82177E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61782E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30167E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44500E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73941E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53661E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52680E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61346E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.39180E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333982 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33396E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33396E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.18825E+03 ;
RUNNING_TIME              (idx, 1)        =  4.21820E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.28492E+00  1.50733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.19185E+02  5.10625E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.46717E-01  1.90833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.66333E-02  1.39167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.02438E+00  4.77167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.21815E+02  7.26440E+02 ];
CPU_USAGE                 (idx, 1)        = 9.92901 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98925E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93554E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.90012E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73567E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.81837E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03572E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40561E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.83005E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56090E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.22425E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78801E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.14221E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.29243E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04698E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85186E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38306E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38435E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35207E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57338E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.85135E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30696E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28202E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.94338E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.50573E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14417E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.94499E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64364E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.27905E-02 -4.90394E+27  2.20079E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03811E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.63308E+18 0.00414  1.74089E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  8.44157E+19 0.00051  8.99891E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.90811E+18 0.00229  5.23218E-02 0.00225 ];
PU239_FISS                (idx, [1:   4]) = [  2.73565E+16 0.02678  2.91611E-04 0.02676 ];
PU240_FISS                (idx, [1:   4]) = [  5.82007E+14 0.21498  6.20641E-06 0.21498 ];
PU241_FISS                (idx, [1:   4]) = [  8.22359E+14 0.17643  8.75915E-06 0.17634 ];
TH232_CAPT                (idx, [1:   4]) = [  9.95680E+19 0.00051  6.85243E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23217E+19 0.00150  8.47997E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  1.72607E+18 0.00361  1.18797E-02 0.00363 ];
U238_CAPT                 (idx, [1:   4]) = [  1.85113E+14 0.37417  1.27537E-06 0.37418 ];
PU239_CAPT                (idx, [1:   4]) = [  1.34728E+16 0.04611  9.26987E-05 0.04606 ];
PU240_CAPT                (idx, [1:   4]) = [  1.90238E+15 0.11189  1.30844E-05 0.11195 ];
PU241_CAPT                (idx, [1:   4]) = [  1.31300E+14 0.44423  9.03574E-07 0.44423 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84137E+17 0.01092  1.26721E-03 0.01089 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001869 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10525E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001869 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5481415 5.48662E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3538950 3.54213E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981504 9.82302E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001869 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.82311E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36143E+20 4.1E-06  2.36143E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36992E+19 1.1E-07  9.36992E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45209E+20 0.00028  1.29030E+20 0.00020  1.61789E+19 0.00167 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38908E+20 0.00017  2.22729E+20 0.00012  1.61789E+19 0.00167 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64833E+20 0.00032  2.64833E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68411E+22 0.00024  3.48892E+22 0.00021  1.95187E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60151E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64923E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20011E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.18553E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.18553E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.13866E-01 0.04739 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.34787E-01 0.01190 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.95879E-04 0.02527 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.36331E+03 0.02792 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01772E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.49298E-01 0.03280 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.75715E-01 0.03280 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52023E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99837E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.92631E-01 0.00043  9.88572E-02 0.00042  3.29975E-04 0.01008 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92378E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.91696E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92378E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89670E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94407E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94389E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44697E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44695E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.85885E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.85960E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.87562E-03 0.00501  2.77371E-04 0.01995  9.12573E-04 0.01090  7.09709E-04 0.01177  1.55448E-03 0.00817  3.44482E-04 0.01843  7.70077E-05 0.03897 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.96558E-01 0.01524  1.24758E-02 4.0E-05  3.22662E-02 0.00013  1.06564E-01 0.00054  3.00149E-01 0.00030  1.25599E+00 0.00077  6.14921E+00 0.02856 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.39040E-03 0.00764  2.43875E-04 0.02800  7.97773E-04 0.01631  6.16070E-04 0.01920  1.36596E-03 0.01257  2.95144E-04 0.02690  7.15729E-05 0.05641 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.08899E-01 0.02318  1.24752E-02 4.4E-05  3.22613E-02 0.00020  1.06613E-01 0.00083  3.00098E-01 0.00045  1.25564E+00 0.00098  6.99241E+00 0.02193 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.51112E-07 0.00127  9.50736E-07 0.00127  1.06538E-06 0.01563 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.48946E-07 0.00120  8.48611E-07 0.00120  9.50798E-07 0.01557 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.33001E-03 0.01024  2.38189E-04 0.03690  7.94809E-04 0.02087  6.15276E-04 0.02187  1.29697E-03 0.01574  3.08795E-04 0.02904  7.59675E-05 0.06126 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.15085E-01 0.02553  1.24740E-02 6.9E-05  3.22665E-02 0.00024  1.06587E-01 0.00099  3.00142E-01 0.00055  1.25218E+00 0.00127  7.05743E+00 0.02940 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.37443E-07 0.01961  8.37046E-07 0.01962  9.38341E-07 0.05385 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.47633E-07 0.01961  7.47278E-07 0.01961  8.37892E-07 0.05392 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09459E-03 0.04576  1.91092E-04 0.16376  6.82184E-04 0.08965  5.59208E-04 0.10055  1.26895E-03 0.06418  3.10534E-04 0.13502  8.26259E-05 0.25606 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.67651E-01 0.12003  1.24734E-02 0.00021  3.22673E-02 0.00077  1.06603E-01 0.00336  3.00271E-01 0.00192  1.25323E+00 0.00366  7.27816E+00 0.08804 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06123E-03 0.04506  1.89266E-04 0.16228  6.62936E-04 0.08550  5.56924E-04 0.09643  1.26521E-03 0.06296  3.09370E-04 0.13415  7.75223E-05 0.25895 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.65995E-01 0.11789  1.24735E-02 0.00021  3.22691E-02 0.00077  1.06609E-01 0.00337  3.00303E-01 0.00191  1.25297E+00 0.00364  7.28997E+00 0.08793 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.71241E+03 0.04184 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.43705E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.42336E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39283E-03 0.00639 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.59531E+03 0.00637 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90654E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63760E-05 0.00849  6.63963E-05 0.00850  1.28499E-06 0.41331 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.64548E-05 0.02543  5.64086E-05 0.02549  1.54553E-06 0.55122 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.86147E-04 0.02519  1.86273E-04 0.02521  1.56106E-04 0.40555 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63283E+01 0.01077 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52680E+01 0.00026  4.00653E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 08:28:35 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01391E+00  9.98043E-01  1.00381E+00  1.00330E+00  9.92609E-01  1.00329E+00  1.00004E+00  9.91349E-01  1.00681E+00  9.95024E-01  1.00423E+00  9.99719E-01  9.95456E-01  1.00454E+00  9.93966E-01  1.00045E+00  9.99469E-01  9.93986E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.82662E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61734E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30216E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44558E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73833E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52781E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51798E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59446E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.39163E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333736 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33413E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33413E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.70556E+03 ;
RUNNING_TIME              (idx, 1)        =  4.73746E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.43168E+00  1.46767E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.70918E+02  5.17327E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.65000E-01  1.82833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.66333E-02  1.39167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.07040E+00  4.59833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.73741E+02  7.29879E+02 ];
CPU_USAGE                 (idx, 1)        = 9.93267 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98938E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.93950E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.92297E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74166E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.25400E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02874E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40035E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85965E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56720E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.21495E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99946E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.03989E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.48079E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13541E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04357E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81913E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38524E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35149E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57322E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90142E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73683E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29077E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99311E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.93972E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.15452E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.93483E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30374E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.80138E-02 -6.02787E+27  2.21202E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03485E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.61238E+18 0.00438  1.72006E-02 0.00430 ];
U233_FISS                 (idx, [1:   4]) = [  8.33189E+19 0.00058  8.88890E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.56841E+18 0.00218  5.94059E-02 0.00209 ];
U238_FISS                 (idx, [1:   4]) = [  2.69626E+13 1.00000  2.81889E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  5.89835E+16 0.02164  6.29357E-04 0.02165 ];
PU240_FISS                (idx, [1:   4]) = [  2.27096E+15 0.10781  2.42331E-05 0.10777 ];
PU241_FISS                (idx, [1:   4]) = [  2.42813E+15 0.09837  2.59345E-05 0.09836 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81247E+19 0.00052  6.76555E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21034E+19 0.00167  8.34494E-02 0.00158 ];
U235_CAPT                 (idx, [1:   4]) = [  1.95611E+18 0.00368  1.34875E-02 0.00369 ];
U238_CAPT                 (idx, [1:   4]) = [  3.68825E+14 0.28013  2.53670E-06 0.28024 ];
PU239_CAPT                (idx, [1:   4]) = [  2.65438E+16 0.03154  1.83087E-04 0.03158 ];
PU240_CAPT                (idx, [1:   4]) = [  5.48917E+15 0.06553  3.78794E-05 0.06552 ];
PU241_CAPT                (idx, [1:   4]) = [  5.55641E+14 0.23192  3.83317E-06 0.23186 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83349E+17 0.01276  1.26433E-03 0.01280 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002398 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10839E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002398 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5478639 5.48358E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3540918 3.54390E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 982841 9.83608E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002398 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.73809E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36137E+20 4.0E-06  2.36137E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36847E+19 9.6E-08  9.36847E+19 9.6E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45114E+20 0.00031  1.28887E+20 0.00022  1.62270E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38798E+20 0.00019  2.22571E+20 0.00013  1.62270E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64494E+20 0.00036  2.64494E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67421E+22 0.00025  3.47869E+22 0.00022  1.95514E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60167E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64815E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19626E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.15129E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.15129E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.48787E-01 0.04972 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.29015E-01 0.01165 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94167E-04 0.02307 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.37909E+03 0.02916 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01641E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.03341E-01 0.03696 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.34222E-01 0.03696 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52055E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99868E+02 9.6E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.93202E-01 0.00046  9.89073E-02 0.00046  3.42850E-04 0.00891 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92704E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92822E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92704E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.90078E-01 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92261E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92422E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47833E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47566E-02 0.00086 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90500E-01 0.00147 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90405E-01 0.00043 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93274E-03 0.00551  3.01385E-04 0.02058  9.14532E-04 0.01146  7.10054E-04 0.01236  1.57202E-03 0.00887  3.55068E-04 0.01692  7.96755E-05 0.03723 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.01467E-01 0.01465  1.24753E-02 3.6E-05  3.22539E-02 0.00013  1.06567E-01 0.00051  3.00369E-01 0.00033  1.25723E+00 0.00069  6.14394E+00 0.02767 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48118E-03 0.00808  2.56978E-04 0.02929  8.20129E-04 0.01534  6.25977E-04 0.01787  1.39063E-03 0.01318  3.14500E-04 0.02586  7.29671E-05 0.06326 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.07166E-01 0.02355  1.24754E-02 4.7E-05  3.22414E-02 0.00018  1.06542E-01 0.00076  3.00198E-01 0.00045  1.25616E+00 0.00095  6.96694E+00 0.02141 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.39777E-07 0.00137  9.39250E-07 0.00137  1.08639E-06 0.01688 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.39356E-07 0.00128  8.38886E-07 0.00128  9.70338E-07 0.01686 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45044E-03 0.00914  2.65838E-04 0.03325  8.20289E-04 0.01860  6.16254E-04 0.02258  1.38338E-03 0.01489  2.94835E-04 0.03190  6.98516E-05 0.05953 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.91544E-01 0.02487  1.24760E-02 5.4E-05  3.22552E-02 0.00021  1.06688E-01 0.00100  3.00280E-01 0.00052  1.25661E+00 0.00136  6.66646E+00 0.03277 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.30548E-07 0.01968  8.30133E-07 0.01968  9.49558E-07 0.06235 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.41908E-07 0.01968  7.41537E-07 0.01969  8.48360E-07 0.06261 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13533E-03 0.04701  2.00067E-04 0.14035  8.35231E-04 0.07771  4.82784E-04 0.09526  1.28091E-03 0.06750  2.65325E-04 0.15760  7.10128E-05 0.31238 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42549E-01 0.08297  1.24777E-02 0.00012  3.22351E-02 0.00061  1.07013E-01 0.00378  3.00721E-01 0.00197  1.25339E+00 0.00366  6.80133E+00 0.11398 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13243E-03 0.04669  2.02512E-04 0.13698  8.40354E-04 0.07625  4.87088E-04 0.09444  1.27706E-03 0.06633  2.57991E-04 0.15554  6.74275E-05 0.30968 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.37684E-01 0.08051  1.24776E-02 0.00012  3.22337E-02 0.00062  1.07023E-01 0.00378  3.00554E-01 0.00193  1.25303E+00 0.00363  6.80133E+00 0.11398 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79977E+03 0.04340 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.33989E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.34192E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53812E-03 0.00590 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.78880E+03 0.00594 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89592E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68718E-05 0.00796  6.68486E-05 0.00799  2.38740E-06 0.31751 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65680E-05 0.02630  5.65158E-05 0.02629  2.33685E-06 0.51306 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84284E-04 0.02273  1.84007E-04 0.02286  2.66719E-04 0.31286 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62033E+01 0.01134 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51798E+01 0.00027  3.98517E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 09:21:15 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00975E+00  9.99201E-01  1.00072E+00  1.00493E+00  9.94141E-01  9.97345E-01  1.00140E+00  9.97159E-01  1.00061E+00  9.96997E-01  9.96345E-01  1.00032E+00  1.00070E+00  1.00364E+00  9.95199E-01  1.00685E+00  9.96698E-01  9.97987E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.83074E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61693E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30282E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44629E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73733E+00 0.00027  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52451E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51473E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.58555E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.39334E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334308 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33408E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33408E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.23020E+03 ;
RUNNING_TIME              (idx, 1)        =  5.26415E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.58725E+00  1.55567E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.23381E+02  5.24628E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.84217E-01  1.92167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.66333E-02  1.39167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.12067E+00  5.02167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.26410E+02  7.34040E+02 ];
CPU_USAGE                 (idx, 1)        = 9.93551 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98961E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.94265E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.91921E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74165E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.86378E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01147E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38842E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87319E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56841E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.32960E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17935E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.07499E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.64655E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21213E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20632E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20350E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37633E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34355E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56298E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92258E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11866E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28045E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01317E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.28222E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.12808E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.94023E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96384E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.33035E-02 -7.16606E+27  2.22341E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03377E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.59132E+18 0.00406  1.69787E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  8.25158E+19 0.00058  8.80411E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.03845E+18 0.00216  6.44290E-02 0.00212 ];
U238_FISS                 (idx, [1:   4]) = [  5.29716E+13 0.70592  5.69342E-07 0.70593 ];
PU239_FISS                (idx, [1:   4]) = [  1.03009E+17 0.01566  1.09913E-03 0.01567 ];
PU240_FISS                (idx, [1:   4]) = [  4.89955E+15 0.07651  5.23024E-05 0.07656 ];
PU241_FISS                (idx, [1:   4]) = [  5.87591E+15 0.07223  6.26707E-05 0.07219 ];
TH232_CAPT                (idx, [1:   4]) = [  9.72992E+19 0.00054  6.69500E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20175E+19 0.00145  8.26904E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  2.12272E+18 0.00354  1.46062E-02 0.00352 ];
U238_CAPT                 (idx, [1:   4]) = [  4.22163E+14 0.24368  2.90544E-06 0.24367 ];
PU239_CAPT                (idx, [1:   4]) = [  4.69414E+16 0.02274  3.22941E-04 0.02271 ];
PU240_CAPT                (idx, [1:   4]) = [  1.24849E+16 0.04582  8.58966E-05 0.04581 ];
PU241_CAPT                (idx, [1:   4]) = [  1.16736E+15 0.15061  8.02427E-06 0.15045 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83571E+17 0.01181  1.26313E-03 0.01181 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002225 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11779E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002225 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5485814 5.49096E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3538152 3.54113E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 978259 9.79087E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002225 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.16067E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36157E+20 4.1E-06  2.36157E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36719E+19 9.5E-08  9.36719E+19 9.5E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45350E+20 0.00028  1.29004E+20 0.00020  1.63457E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39021E+20 0.00017  2.22676E+20 0.00012  1.63457E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64674E+20 0.00035  2.64674E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67539E+22 0.00025  3.47901E+22 0.00022  1.96371E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59146E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64936E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19621E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.11847E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.11847E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.59632E-01 0.04696 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21755E-01 0.01223 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.99059E-04 0.02203 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.15790E+03 0.02920 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02094E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.22610E-01 0.03517 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.51741E-01 0.03518 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52111E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99895E+02 9.5E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.92666E-01 0.00047  9.88465E-02 0.00046  3.44793E-04 0.00920 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92370E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92286E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92370E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89237E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90540E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90366E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50400E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50636E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.93490E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.94486E-01 0.00043 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.98945E-03 0.00516  2.92757E-04 0.01977  9.30723E-04 0.01115  7.15963E-04 0.01337  1.61606E-03 0.00850  3.59178E-04 0.01884  7.47694E-05 0.03966 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.91181E-01 0.01425  1.24763E-02 5.6E-05  3.22394E-02 0.00013  1.06626E-01 0.00056  3.00385E-01 0.00027  1.25832E+00 0.00083  6.08954E+00 0.02903 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49142E-03 0.00729  2.61066E-04 0.02906  8.11018E-04 0.01716  6.27784E-04 0.01919  1.41573E-03 0.01193  3.10469E-04 0.02634  6.53466E-05 0.05428 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.91384E-01 0.02006  1.24768E-02 7.1E-05  3.22357E-02 0.00018  1.06590E-01 0.00079  3.00518E-01 0.00041  1.25803E+00 0.00113  6.87436E+00 0.02249 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.34532E-07 0.00125  9.34175E-07 0.00125  1.03929E-06 0.01748 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.34177E-07 0.00119  8.33859E-07 0.00119  9.27500E-07 0.01744 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46658E-03 0.00940  2.53387E-04 0.03630  8.10439E-04 0.01850  6.33230E-04 0.02222  1.40474E-03 0.01340  3.01834E-04 0.02957  6.29567E-05 0.06837 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.87796E-01 0.02500  1.24752E-02 6.3E-05  3.22354E-02 0.00023  1.06670E-01 0.00109  3.00414E-01 0.00049  1.25704E+00 0.00145  7.16796E+00 0.03006 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.22587E-07 0.01974  8.22263E-07 0.01974  8.73389E-07 0.05478 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.34395E-07 0.01973  7.34103E-07 0.01973  7.80063E-07 0.05484 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08395E-03 0.04493  2.25777E-04 0.16232  7.03031E-04 0.08287  5.78736E-04 0.09736  1.25189E-03 0.06402  2.84280E-04 0.12272  4.02342E-05 0.25190 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.00339E-01 0.10254  1.24742E-02 0.00019  3.22381E-02 0.00082  1.06585E-01 0.00338  3.00123E-01 0.00183  1.25947E+00 0.00370  8.77789E+00 0.05227 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09140E-03 0.04442  2.27052E-04 0.15526  7.19287E-04 0.08219  5.57982E-04 0.09571  1.26584E-03 0.06339  2.84447E-04 0.12196  3.67887E-05 0.25145 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.95649E-01 0.10058  1.24743E-02 0.00019  3.22371E-02 0.00081  1.06633E-01 0.00339  3.00121E-01 0.00182  1.25954E+00 0.00371  8.73662E+00 0.05322 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.78775E+03 0.04151 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.26011E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.26569E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.52388E-03 0.00597 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.80740E+03 0.00620 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89028E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75191E-05 0.00714  6.74642E-05 0.00717  2.10939E-06 0.35785 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.59723E-05 0.02592  5.58818E-05 0.02606  1.60021E-06 0.38282 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.91439E-04 0.02157  1.91403E-04 0.02169  1.98806E-04 0.34963 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62081E+01 0.01143 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51473E+01 0.00027  3.97103E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 10:14:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00442E+00  9.93904E-01  1.00016E+00  1.00740E+00  9.97589E-01  9.99142E-01  9.98559E-01  9.97417E-01  1.00510E+00  9.98966E-01  9.99059E-01  9.98118E-01  9.92449E-01  1.00572E+00  9.97515E-01  9.99725E-01  9.99647E-01  1.00511E+00  ];
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

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.6E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.86404E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.61360E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29651E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44109E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73954E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50918E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49937E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57611E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.41567E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333689 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33387E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33387E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.76033E+03 ;
RUNNING_TIME              (idx, 1)        =  5.79635E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.74150E+00  1.54250E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.76393E+02  5.30122E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.02817E-01  1.86000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.66333E-02  1.39167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.16778E+00  4.70667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.79630E+02  7.37537E+02 ];
CPU_USAGE                 (idx, 1)        = 9.93786 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98959E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.94525E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.94075E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74775E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.13623E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97605E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36753E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.93015E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57662E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.81369E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75858E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.32328E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04503E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44269E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70414E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34182E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37262E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33812E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55704E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.96797E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27197E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27686E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.06564E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.42073E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10477E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.95697E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60423E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.42746E-02 -1.16785E+28  2.26853E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03122E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.56767E+18 0.00384  1.67439E-02 0.00377 ];
U233_FISS                 (idx, [1:   4]) = [  8.05073E+19 0.00051  8.59919E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.90393E+18 0.00218  7.37392E-02 0.00205 ];
U238_FISS                 (idx, [1:   4]) = [  1.33354E+14 0.44424  1.42390E-06 0.44428 ];
PU239_FISS                (idx, [1:   4]) = [  3.31394E+17 0.00917  3.53966E-03 0.00916 ];
PU240_FISS                (idx, [1:   4]) = [  2.60292E+16 0.03322  2.78101E-04 0.03323 ];
PU241_FISS                (idx, [1:   4]) = [  3.87604E+16 0.02625  4.13862E-04 0.02621 ];
TH232_CAPT                (idx, [1:   4]) = [  9.51520E+19 0.00056  6.52394E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17300E+19 0.00148  8.04256E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.44692E+18 0.00338  1.67767E-02 0.00334 ];
U238_CAPT                 (idx, [1:   4]) = [  8.97948E+14 0.17176  6.16215E-06 0.17185 ];
PU239_CAPT                (idx, [1:   4]) = [  1.55449E+17 0.01359  1.06586E-03 0.01361 ];
PU240_CAPT                (idx, [1:   4]) = [  6.83039E+16 0.02049  4.68220E-04 0.02044 ];
PU241_CAPT                (idx, [1:   4]) = [  8.70782E+15 0.05549  5.96796E-05 0.05548 ];
SM149_CAPT                (idx, [1:   4]) = [  1.91442E+17 0.01186  1.31261E-03 0.01186 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001602 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07413E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001602 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5493803 5.49899E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3526726 3.52986E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981073 9.81886E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001602 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -9.49949E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36329E+20 4.6E-06  2.36329E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36324E+19 1.0E-07  9.36324E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45870E+20 0.00030  1.29471E+20 0.00020  1.63989E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39502E+20 0.00018  2.23103E+20 0.00012  1.63989E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65232E+20 0.00035  2.65232E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68413E+22 0.00025  3.48773E+22 0.00022  1.96399E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.60435E+19 0.00136 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65546E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19460E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.99346E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.99346E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88328E-01 0.04498 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30135E-01 0.01168 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.20110E-04 0.02284 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.78820E+03 0.02938 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01814E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.76938E-01 0.03012 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.00678E-01 0.03012 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52401E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99979E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.90922E-01 0.00043  9.86530E-02 0.00043  3.39169E-04 0.00841 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90952E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.91058E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90952E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87948E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.86158E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.86189E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.57137E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.57061E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.03721E-01 0.00141 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.04019E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.98973E-03 0.00534  2.84765E-04 0.02110  9.31317E-04 0.01105  7.08177E-04 0.01237  1.61339E-03 0.00791  3.71238E-04 0.01753  8.08403E-05 0.03665 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.11855E-01 0.01430  1.24773E-02 6.2E-05  3.22194E-02 0.00014  1.06883E-01 0.00058  3.01237E-01 0.00032  1.25800E+00 0.00090  6.65188E+00 0.02396 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46645E-03 0.00762  2.51652E-04 0.03021  8.28845E-04 0.01647  5.97221E-04 0.01927  1.39445E-03 0.01121  3.23295E-04 0.02499  7.09933E-05 0.05703 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.14607E-01 0.02160  1.24776E-02 6.9E-05  3.22221E-02 0.00019  1.06840E-01 0.00076  3.01233E-01 0.00045  1.25787E+00 0.00113  7.14014E+00 0.02019 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.15898E-07 0.00136  9.15476E-07 0.00136  1.03988E-06 0.01976 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.15939E-07 0.00127  8.15563E-07 0.00126  9.26443E-07 0.01978 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41860E-03 0.00844  2.44971E-04 0.03404  7.93250E-04 0.01744  6.06692E-04 0.02093  1.38389E-03 0.01430  3.19027E-04 0.02837  7.07711E-05 0.06277 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14614E-01 0.02463  1.24779E-02 0.00013  3.22196E-02 0.00027  1.06923E-01 0.00105  3.01236E-01 0.00054  1.25705E+00 0.00184  7.07913E+00 0.02997 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.05076E-07 0.01964  8.04814E-07 0.01964  8.97228E-07 0.05604 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.17327E-07 0.01963  7.17091E-07 0.01963  8.00122E-07 0.05607 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14508E-03 0.04209  2.65161E-04 0.13778  7.41604E-04 0.07676  6.24334E-04 0.08305  1.21547E-03 0.06155  2.42258E-04 0.13232  5.62539E-05 0.24897 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.61055E-01 0.08146  1.24761E-02 0.00016  3.22340E-02 0.00078  1.07261E-01 0.00370  3.00979E-01 0.00200  1.26202E+00 0.00569  6.52292E+00 0.09571 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13315E-03 0.04178  2.51254E-04 0.13354  7.30536E-04 0.07577  6.26357E-04 0.08417  1.21369E-03 0.06024  2.52133E-04 0.13169  5.91783E-05 0.24777 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.71340E-01 0.08266  1.24760E-02 0.00016  3.22321E-02 0.00078  1.07226E-01 0.00367  3.00964E-01 0.00198  1.26227E+00 0.00569  6.52337E+00 0.09570 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.92017E+03 0.03725 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.07728E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08675E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.49175E-03 0.00553 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84718E+03 0.00556 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92716E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75756E-05 0.00646  6.76235E-05 0.00646  1.58778E-06 0.34108 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.74705E-05 0.02323  5.74600E-05 0.02324  1.63937E-06 0.42723 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.09895E-04 0.02262  2.09832E-04 0.02254  2.25211E-04 0.33156 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59520E+01 0.01114 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49937E+01 0.00026  3.92837E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 11:07:54 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01178E+00  9.91848E-01  1.00280E+00  1.00225E+00  9.93710E-01  9.99747E-01  1.00147E+00  1.00559E+00  9.95567E-01  1.00185E+00  9.97508E-01  9.96523E-01  1.00175E+00  1.00281E+00  9.95435E-01  1.00156E+00  9.99335E-01  9.98478E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.93318E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.60668E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.28394E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.43068E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75499E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50659E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49666E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61059E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.48005E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333219 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33369E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33369E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.29250E+03 ;
RUNNING_TIME              (idx, 1)        =  6.33060E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.89647E+00  1.54967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.29609E+02  5.32159E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.21717E-01  1.89000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.06500E-02  1.40167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.21330E+00  4.54667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.33056E+02  7.39428E+02 ];
CPU_USAGE                 (idx, 1)        = 9.93982 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98956E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.94744E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.92321E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74180E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.01197E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92989E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34289E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95878E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57313E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.49214E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38378E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.31752E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13446E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69254E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23874E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57871E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34635E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31715E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52889E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00048E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59817E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24070E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04173E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.10109E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.03238E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.98553E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32052E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.09027E-01 -2.34598E+28  2.38634E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02449E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.52034E+18 0.00405  1.62346E-02 0.00399 ];
U233_FISS                 (idx, [1:   4]) = [  7.93670E+19 0.00052  8.47546E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.13142E+18 0.00197  7.61531E-02 0.00186 ];
U238_FISS                 (idx, [1:   4]) = [  1.86810E+14 0.37418  1.99795E-06 0.37420 ];
PU239_FISS                (idx, [1:   4]) = [  6.25942E+17 0.00667  6.68358E-03 0.00660 ];
PU240_FISS                (idx, [1:   4]) = [  7.35499E+16 0.01865  7.85423E-04 0.01863 ];
PU241_FISS                (idx, [1:   4]) = [  1.31101E+17 0.01390  1.39983E-03 0.01387 ];
TH232_CAPT                (idx, [1:   4]) = [  9.33252E+19 0.00055  6.37497E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15836E+19 0.00141  7.91274E-02 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49452E+18 0.00348  1.70404E-02 0.00348 ];
U238_CAPT                 (idx, [1:   4]) = [  2.87645E+15 0.09774  1.96387E-05 0.09777 ];
PU239_CAPT                (idx, [1:   4]) = [  2.85390E+17 0.00970  1.94945E-03 0.00969 ];
PU240_CAPT                (idx, [1:   4]) = [  1.97498E+17 0.01131  1.34926E-03 0.01134 ];
PU241_CAPT                (idx, [1:   4]) = [  2.76096E+16 0.03239  1.88597E-04 0.03236 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90728E+17 0.01196  1.30285E-03 0.01194 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001060 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07997E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001060 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494169 5.49971E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3514705 3.51802E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 992186 9.93062E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001060 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -6.14673E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36623E+20 5.0E-06  2.36623E+20 5.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35944E+19 1.1E-07  9.35944E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46383E+20 0.00031  1.29433E+20 0.00020  1.69500E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39977E+20 0.00019  2.23027E+20 0.00012  1.69500E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66184E+20 0.00033  2.66184E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.71983E+22 0.00024  3.51902E+22 0.00021  2.00806E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.64341E+19 0.00135 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66411E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19818E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.74475E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.74475E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.15527E-01 0.03857 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.18037E-01 0.00995 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.83426E-04 0.01938 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.45492E+03 0.03056 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.00698E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.26168E-01 0.02556 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.44140E-01 0.02556 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52818E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00060E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.89209E-01 0.00043  9.84830E-02 0.00042  3.45039E-04 0.00889 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.89161E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88975E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.89161E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87219E-01 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83588E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83379E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61220E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61534E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10186E-01 0.00130 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10530E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.00864E-03 0.00528  2.87826E-04 0.02006  9.34463E-04 0.01122  6.97874E-04 0.01312  1.63298E-03 0.00862  3.72147E-04 0.01649  8.33501E-05 0.03565 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.15050E-01 0.01328  1.24807E-02 0.00014  3.22040E-02 0.00016  1.06665E-01 0.00052  3.01244E-01 0.00030  1.26049E+00 0.00101  6.75961E+00 0.02374 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45544E-03 0.00816  2.39771E-04 0.02900  7.97893E-04 0.01702  5.95926E-04 0.01810  1.41499E-03 0.01284  3.31352E-04 0.02690  7.55090E-05 0.05019 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.25051E-01 0.01811  1.24797E-02 0.00012  3.22015E-02 0.00023  1.06715E-01 0.00081  3.01273E-01 0.00043  1.25818E+00 0.00151  7.28851E+00 0.01976 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.14055E-07 0.00152  9.13596E-07 0.00153  1.04098E-06 0.01773 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.12724E-07 0.00141  8.12317E-07 0.00142  9.25516E-07 0.01771 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48276E-03 0.00905  2.54282E-04 0.03416  8.03512E-04 0.01822  6.08100E-04 0.02136  1.41887E-03 0.01459  3.28092E-04 0.02923  6.99028E-05 0.06235 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.12825E-01 0.02265  1.24819E-02 0.00025  3.21981E-02 0.00027  1.06551E-01 0.00094  3.01268E-01 0.00051  1.25995E+00 0.00184  7.43073E+00 0.02764 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.01349E-07 0.01968  8.01097E-07 0.01968  8.67286E-07 0.06344 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.12698E-07 0.01968  7.12473E-07 0.01969  7.71391E-07 0.06338 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00226E-03 0.04314  2.28335E-04 0.13529  6.42499E-04 0.08269  5.90942E-04 0.09717  1.16953E-03 0.06213  3.22414E-04 0.12817  4.85423E-05 0.25566 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52282E-01 0.06481  1.24889E-02 0.00082  3.21736E-02 0.00087  1.07206E-01 0.00351  3.00810E-01 0.00200  1.25971E+00 0.00501  7.42253E+00 0.08872 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00139E-03 0.04253  2.32581E-04 0.13246  6.32644E-04 0.08144  5.94940E-04 0.09505  1.18064E-03 0.06108  3.12915E-04 0.12404  4.76714E-05 0.25741 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.56010E-01 0.06939  1.24889E-02 0.00082  3.21752E-02 0.00085  1.07186E-01 0.00350  3.00751E-01 0.00198  1.25963E+00 0.00501  7.42253E+00 0.08872 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.76421E+03 0.03851 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.05270E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.04927E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.49301E-03 0.00557 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.85959E+03 0.00568 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.07026E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.79288E-05 0.00587  6.79105E-05 0.00589  2.17130E-06 0.33417 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88484E-05 0.01991  5.87905E-05 0.02001  2.22529E-06 0.42476 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.70003E-04 0.01917  2.70178E-04 0.01932  2.14098E-04 0.32926 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62143E+01 0.01117 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49666E+01 0.00028  3.90292E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 12:01:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00791E+00  1.00077E+00  1.00670E+00  9.99076E-01  9.97341E-01  1.00106E+00  1.00147E+00  9.93450E-01  9.95562E-01  9.99752E-01  9.96351E-01  1.00297E+00  1.00069E+00  9.99546E-01  1.00412E+00  9.98953E-01  9.95420E-01  9.98855E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.01911E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59809E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.26952E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.41898E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77117E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51335E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50330E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.66822E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.56363E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333663 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33413E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33413E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.82700E+03 ;
RUNNING_TIME              (idx, 1)        =  6.86728E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.05433E+00  1.57867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.83058E+02  5.34487E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.41617E-01  1.99000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.06500E-02  1.40167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.26002E+00  4.66500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.86723E+02  7.40117E+02 ];
CPU_USAGE                 (idx, 1)        = 9.94135 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98953E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.94922E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.91836E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74012E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.38981E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91583E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33502E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96785E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57210E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57416E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55486E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39193E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14709E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76509E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39649E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93125E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33991E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31248E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52230E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01833E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00924E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22905E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03173E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27027E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00880E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.99303E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98062E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.64903E-01 -3.54829E+28  2.50658E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01925E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.54071E+18 0.00399  1.64604E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  7.92073E+19 0.00060  8.46172E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.08197E+18 0.00189  7.56580E-02 0.00184 ];
U238_FISS                 (idx, [1:   4]) = [  2.67806E+14 0.31146  2.85323E-06 0.31145 ];
PU239_FISS                (idx, [1:   4]) = [  6.54045E+17 0.00634  6.98717E-03 0.00632 ];
PU240_FISS                (idx, [1:   4]) = [  8.57536E+16 0.01796  9.15764E-04 0.01786 ];
PU241_FISS                (idx, [1:   4]) = [  1.56206E+17 0.01265  1.66903E-03 0.01269 ];
TH232_CAPT                (idx, [1:   4]) = [  9.25639E+19 0.00053  6.32585E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.14982E+19 0.00156  7.85796E-02 0.00151 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49330E+18 0.00305  1.70399E-02 0.00306 ];
U238_CAPT                 (idx, [1:   4]) = [  3.22173E+15 0.08520  2.20162E-05 0.08518 ];
PU239_CAPT                (idx, [1:   4]) = [  3.06258E+17 0.00914  2.09304E-03 0.00914 ];
PU240_CAPT                (idx, [1:   4]) = [  2.24620E+17 0.01095  1.53531E-03 0.01099 ];
PU241_CAPT                (idx, [1:   4]) = [  3.26740E+16 0.02799  2.23350E-04 0.02803 ];
SM149_CAPT                (idx, [1:   4]) = [  2.00012E+17 0.01170  1.36684E-03 0.01168 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002395 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06252E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002395 1.00106E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487351 5.49208E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3510526 3.51332E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1004518 1.00523E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002395 1.00106E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.14204E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36678E+20 4.3E-06  2.36678E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35893E+19 1.1E-07  9.35893E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46400E+20 0.00028  1.28932E+20 0.00019  1.74672E+19 0.00159 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39989E+20 0.00017  2.22522E+20 0.00011  1.74672E+19 0.00159 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66434E+20 0.00036  2.66434E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.75408E+22 0.00025  3.55068E+22 0.00022  2.03396E+21 0.00161 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.67837E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66773E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20105E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.53904E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.53904E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.03491E-01 0.03801 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07903E-01 0.00837 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.80967E-04 0.01699 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.97472E+03 0.03615 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99484E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.39577E-01 0.02430 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.55063E-01 0.02430 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52890E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.88438E-01 0.00045  9.83799E-02 0.00044  3.39280E-04 0.00982 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.88154E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88351E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.88154E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87391E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83575E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83444E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61241E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61431E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11078E-01 0.00136 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11321E-01 0.00042 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.01284E-03 0.00532  2.88992E-04 0.01968  9.28704E-04 0.01109  7.22091E-04 0.01320  1.61515E-03 0.00794  3.73681E-04 0.01610  8.42228E-05 0.04063 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.11674E-01 0.01576  1.24805E-02 0.00012  3.22091E-02 0.00016  1.06782E-01 0.00052  3.01467E-01 0.00031  1.25771E+00 0.00096  6.03187E+00 0.02967 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49322E-03 0.00769  2.47777E-04 0.02815  8.12781E-04 0.01617  6.33408E-04 0.01963  1.40564E-03 0.01151  3.21758E-04 0.02382  7.18575E-05 0.06023 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.08153E-01 0.02250  1.24795E-02 0.00014  3.22177E-02 0.00022  1.06850E-01 0.00077  3.01517E-01 0.00041  1.25740E+00 0.00121  6.92672E+00 0.02223 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.28246E-07 0.00157  9.27788E-07 0.00157  1.06509E-06 0.01922 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.24629E-07 0.00147  8.24222E-07 0.00148  9.46373E-07 0.01924 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44576E-03 0.00989  2.42229E-04 0.03621  7.92663E-04 0.01936  6.30004E-04 0.02422  1.39074E-03 0.01576  3.18431E-04 0.02790  7.17001E-05 0.06594 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14151E-01 0.02701  1.24805E-02 0.00015  3.22062E-02 0.00031  1.06864E-01 0.00113  3.01344E-01 0.00055  1.25758E+00 0.00178  6.84877E+00 0.03093 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.10492E-07 0.01964  8.09799E-07 0.01964  9.75094E-07 0.05350 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20136E-07 0.01964  7.19520E-07 0.01964  8.66328E-07 0.05350 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13027E-03 0.04620  3.09592E-04 0.13144  6.69225E-04 0.08520  5.86015E-04 0.10115  1.23568E-03 0.06414  2.55041E-04 0.14281  7.47163E-05 0.24873 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.93492E-01 0.09495  1.24773E-02 0.00012  3.22447E-02 0.00102  1.07104E-01 0.00349  3.01564E-01 0.00206  1.23648E+00 0.00775  7.24722E+00 0.07543 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10999E-03 0.04580  3.10114E-04 0.13051  6.57819E-04 0.08297  5.85955E-04 0.10007  1.22059E-03 0.06279  2.57979E-04 0.14302  7.75354E-05 0.23973 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.05452E-01 0.10009  1.24773E-02 0.00012  3.22418E-02 0.00102  1.07088E-01 0.00348  3.01634E-01 0.00204  1.23653E+00 0.00779  7.26364E+00 0.07519 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88606E+03 0.04208 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.17583E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.15165E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44600E-03 0.00620 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.75558E+03 0.00617 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.27969E-09 0.00095 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.87640E-05 0.00498  6.87943E-05 0.00497  1.39362E-06 0.36227 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.91833E-05 0.01620  5.92010E-05 0.01618  1.31878E-06 0.46594 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.66056E-04 0.01580  3.66729E-04 0.01579  1.96670E-04 0.34997 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59339E+01 0.01105 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50330E+01 0.00027  3.90250E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha2' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 00:34:50 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 12:32:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651203290478 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01569E+00  1.00077E+00  9.99981E-01  9.94812E-01  9.82756E-01  1.00783E+00  9.99006E-01  1.00077E+00  1.00548E+00  1.00164E+00  1.00048E+00  9.88730E-01  9.92053E-01  9.99418E-01  9.98688E-01  9.98355E-01  1.00177E+00  1.01178E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.10740E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58926E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.25643E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.40872E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.78078E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52418E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51408E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.72874E+01 0.00033  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.64954E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333489 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33387E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33387E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.13067E+03 ;
RUNNING_TIME              (idx, 1)        =  7.17305E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.07550E+00  1.07550E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.21238E+00  1.58050E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.13416E+02  3.03578E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.60583E-01  1.89667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  4.06500E-02  1.40167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.30378E+00  4.37167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.17300E+02  7.17300E+02 ];
CPU_USAGE                 (idx, 1)        = 9.94092 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98342E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.94919E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22693.88;
MEMSIZE                   (idx, 1)        = 22534.73;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 383.02;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 159.15;

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

TOT_ACTIVITY              (idx, 1)        =  6.91304E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73944E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.56109E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90698E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32952E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.97141E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57199E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.58318E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60091E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39870E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14580E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78642E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44364E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03265E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33864E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31164E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52104E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02149E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13604E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22229E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02901E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.29461E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99203E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  8.00510E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64071E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.21309E-01 -4.76201E+28  2.62795E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01568E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.53023E+18 0.00397  1.63533E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  7.91706E+19 0.00058  8.46117E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.08329E+18 0.00200  7.57013E-02 0.00193 ];
U238_FISS                 (idx, [1:   4]) = [  2.67005E+14 0.31147  2.86160E-06 0.31148 ];
PU239_FISS                (idx, [1:   4]) = [  6.59150E+17 0.00613  7.04441E-03 0.00609 ];
PU240_FISS                (idx, [1:   4]) = [  8.53777E+16 0.01751  9.12490E-04 0.01750 ];
PU241_FISS                (idx, [1:   4]) = [  1.60590E+17 0.01284  1.71607E-03 0.01280 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21501E+19 0.00059  6.28654E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15113E+19 0.00149  7.85331E-02 0.00149 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48533E+18 0.00346  1.69551E-02 0.00343 ];
U238_CAPT                 (idx, [1:   4]) = [  3.57363E+15 0.08310  2.44054E-05 0.08312 ];
PU239_CAPT                (idx, [1:   4]) = [  3.11001E+17 0.00924  2.12163E-03 0.00923 ];
PU240_CAPT                (idx, [1:   4]) = [  2.30765E+17 0.01067  1.57442E-03 0.01068 ];
PU241_CAPT                (idx, [1:   4]) = [  3.29063E+16 0.02696  2.24473E-04 0.02695 ];
SM149_CAPT                (idx, [1:   4]) = [  2.04035E+17 0.01075  1.39188E-03 0.01073 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001623 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06125E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001623 1.00106E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5488228 5.49338E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3503639 3.50664E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1009756 1.01059E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001623 1.00106E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.33413E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 3.7E-09  3.61697E-02 3.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36686E+20 4.6E-06  2.36686E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35890E+19 1.1E-07  9.35890E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46670E+20 0.00029  1.28613E+20 0.00020  1.80576E+19 0.00160 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40259E+20 0.00018  2.22202E+20 0.00012  1.80576E+19 0.00160 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66837E+20 0.00037  2.66837E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.79148E+22 0.00026  3.58479E+22 0.00025  2.06685E+21 0.00153 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.69676E+19 0.00153 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67227E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20574E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.36458E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.36458E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.00774E-01 0.03179 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13327E-01 0.00769 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.76079E-04 0.01513 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.68678E+03 0.03586 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.98950E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.47045E-01 0.01181 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.51331E-01 0.01181 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52899E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.86736E-01 0.00045  9.81983E-02 0.00045  3.37334E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.86677E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.87044E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.86677E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86317E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83715E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83661E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61025E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61083E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11222E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11404E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.01183E-03 0.00547  2.85276E-04 0.01878  9.31763E-04 0.01070  7.31962E-04 0.01136  1.61439E-03 0.00851  3.62345E-04 0.01858  8.60986E-05 0.03559 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10216E-01 0.01400  1.24782E-02 6.3E-05  3.22030E-02 0.00015  1.06881E-01 0.00053  3.01373E-01 0.00032  1.25496E+00 0.00118  6.40425E+00 0.02511 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46861E-03 0.00766  2.41136E-04 0.02706  8.03822E-04 0.01577  6.39072E-04 0.01803  1.39428E-03 0.01168  3.11601E-04 0.02615  7.87030E-05 0.05091 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.17555E-01 0.02010  1.24781E-02 5.5E-05  3.21986E-02 0.00021  1.06858E-01 0.00078  3.01392E-01 0.00042  1.25556E+00 0.00141  6.94259E+00 0.02114 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.40065E-07 0.00153  9.39705E-07 0.00154  1.05327E-06 0.01827 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.33539E-07 0.00147  8.33221E-07 0.00147  9.33814E-07 0.01825 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41928E-03 0.00900  2.40121E-04 0.03317  8.10462E-04 0.01911  6.13455E-04 0.02104  1.37466E-03 0.01465  3.02368E-04 0.02932  7.82204E-05 0.05847 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14342E-01 0.02297  1.24785E-02 0.00013  3.21942E-02 0.00026  1.06736E-01 0.00090  3.01496E-01 0.00055  1.25410E+00 0.00256  6.78970E+00 0.02949 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.09660E-07 0.01966  8.09343E-07 0.01966  8.65267E-07 0.05473 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.18110E-07 0.01965  7.17829E-07 0.01965  7.67571E-07 0.05474 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.14518E-03 0.04200  2.40416E-04 0.14318  8.65009E-04 0.08066  4.69063E-04 0.10078  1.26434E-03 0.06235  2.36395E-04 0.13844  6.99572E-05 0.29469 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.76500E-01 0.09230  1.24782E-02 0.00011  3.21911E-02 0.00084  1.07265E-01 0.00369  3.01651E-01 0.00204  1.25607E+00 0.00812  6.30674E+00 0.09940 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13319E-03 0.04133  2.43342E-04 0.14167  8.60950E-04 0.07982  4.79046E-04 0.09804  1.23905E-03 0.06137  2.39733E-04 0.13299  7.10702E-05 0.29594 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.82193E-01 0.09245  1.24783E-02 0.00011  3.21908E-02 0.00085  1.07270E-01 0.00368  3.01734E-01 0.00204  1.25646E+00 0.00810  6.30674E+00 0.09940 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.90364E+03 0.03768 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.26613E-07 0.00089 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.21596E-07 0.00068 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44780E-03 0.00509 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.72210E+03 0.00522 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.50504E-09 0.00109 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.89203E-05 0.00415  6.89099E-05 0.00415  2.47670E-06 0.32084 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.23032E-05 0.01554  6.23673E-05 0.01554  1.10256E-06 0.43011 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.53616E-04 0.01481  4.54445E-04 0.01486  2.59038E-04 0.31301 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59000E+01 0.01008 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51408E+01 0.00028  3.90420E+01 0.00038 ];

