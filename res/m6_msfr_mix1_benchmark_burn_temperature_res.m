
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 05:29:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.05736E+00  9.95676E-01  9.92540E-01  9.99855E-01  1.00007E+00  9.87573E-01  9.95431E-01  1.00380E+00  9.95372E-01  9.95377E-01  9.92085E-01  9.95206E-01  9.98483E-01  9.97645E-01  9.86823E-01  1.00433E+00  1.00812E+00  9.94260E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.58300E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74170E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40333E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54174E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59594E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.42333E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.41401E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.04889E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.08603E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333400 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33359E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33359E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.40640E+02 ;
RUNNING_TIME              (idx, 1)        =  2.75644E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.03933E-01  1.03933E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.65015E+01  2.65015E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.91862E+00  2.86395E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.47506E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 8.73012 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98541E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.60710E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  2.84851E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.24012E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.12901E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.84851E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.24012E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.73771E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.45330E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.73771E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.45330E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29315E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.12804E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.30093E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.10748E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20796E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.70533E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.53032E+18 0.00397  1.63178E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  9.22485E+19 0.00049  9.83682E-01 6.5E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.19583E+19 0.00057  7.58018E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33825E+19 0.00135  1.10313E-01 0.00127 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000773 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15171E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000773 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5114971 5.12056E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3954077 3.95836E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931725 9.32602E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000773 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.15484E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36473E+20 3.8E-06  2.36473E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37765E+19 1.3E-07  9.37765E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21277E+20 0.00031  1.08392E+20 0.00023  1.28853E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15054E+20 0.00017  2.02168E+20 0.00012  1.28853E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.36916E+20 0.00036  2.36916E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.10546E+22 0.00025  2.94845E+22 0.00023  1.57006E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20953E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.37149E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.04689E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51105E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.25537E-01 0.08605 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.91044E-01 0.00891 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.23151E-04 0.01557 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.17965E+04 0.02352 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06749E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.32554E-01 0.07183 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.92213E-01 0.07183 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52166E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.98226E-01 0.00043  1.10556E-01 0.00043  3.51490E-04 0.00883 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.98304E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.98168E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.98304E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.10100E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.99319E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  6.99496E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.37751E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.37492E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50554E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50411E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.21980E-03 0.00555  2.61341E-04 0.02080  7.98062E-04 0.01041  5.85731E-04 0.01380  1.26001E-03 0.00871  2.62240E-04 0.02021  5.24150E-05 0.04474 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.48335E-01 0.01561  1.24760E-02 2.7E-05  3.23385E-02 9.1E-05  1.06017E-01 0.00058  2.97227E-01 0.00024  1.23508E+00 0.00021  5.09849E+00 0.03980 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21137E-03 0.00787  2.67735E-04 0.03301  7.86701E-04 0.01572  5.81067E-04 0.01818  1.26083E-03 0.01314  2.67919E-04 0.03002  4.71213E-05 0.07153 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.40928E-01 0.02426  1.24760E-02 3.6E-05  3.23372E-02 0.00012  1.05925E-01 0.00075  2.97157E-01 0.00036  1.23520E+00 0.00030  6.34903E+00 0.03037 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.00826E-07 0.00121  8.00573E-07 0.00121  8.81890E-07 0.01523 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.99367E-07 0.00115  7.99115E-07 0.00116  8.80260E-07 0.01523 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.17258E-03 0.00900  2.57985E-04 0.03128  7.79648E-04 0.01708  5.91376E-04 0.02219  1.24456E-03 0.01411  2.50775E-04 0.03308  4.82347E-05 0.07231 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.41591E-01 0.02484  1.24769E-02 3.4E-05  3.23384E-02 0.00015  1.06014E-01 0.00101  2.97144E-01 0.00040  1.23462E+00 0.00044  6.42761E+00 0.04132 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.09169E-07 0.01967  7.09053E-07 0.01968  7.18835E-07 0.05269 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.08087E-07 0.01967  7.07971E-07 0.01968  7.17783E-07 0.05273 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.97354E-03 0.04379  2.35690E-04 0.12829  6.96187E-04 0.07927  5.84181E-04 0.09605  1.18299E-03 0.06720  2.12655E-04 0.14335  6.18389E-05 0.31101 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.84380E-01 0.12393  1.24750E-02 0.00016  3.23411E-02 0.00054  1.06413E-01 0.00340  2.97782E-01 0.00167  1.23500E+00 0.00116  7.74812E+00 0.09706 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.95795E-03 0.04325  2.33700E-04 0.12373  7.00845E-04 0.07789  5.79624E-04 0.09364  1.17221E-03 0.06796  2.12956E-04 0.13483  5.86173E-05 0.30847 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.81861E-01 0.12183  1.24751E-02 0.00015  3.23398E-02 0.00053  1.06479E-01 0.00344  2.97747E-01 0.00163  1.23497E+00 0.00116  7.73717E+00 0.09706 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.19877E+03 0.03973 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  7.93652E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.92207E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17601E-03 0.00557 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.00174E+03 0.00554 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.04679E-09 0.00122 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.22569E-05 0.00423  7.22388E-05 0.00424  3.13035E-06 0.28912 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.74215E-05 0.01587  7.72543E-05 0.01593  5.32089E-06 0.32510 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.16183E-04 0.01512  4.16330E-04 0.01516  3.78838E-04 0.28440 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73985E+01 0.01114 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.41401E+01 0.00026  3.98494E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 06:11:20 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01890E+00  9.93703E-01  9.97186E-01  1.00439E+00  1.00496E+00  1.00171E+00  1.00914E+00  1.00243E+00  1.00007E+00  1.00781E+00  9.95466E-01  9.96162E-01  1.00994E+00  1.00704E+00  1.00411E+00  1.00172E+00  1.00692E+00  9.38360E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.57541E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74246E-01 8.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40399E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54200E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59712E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.43619E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.42694E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.07151E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.07317E-01 0.00058  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333719 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33385E+04 0.00052 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33385E+04 0.00052 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.11446E+02 ;
RUNNING_TIME              (idx, 1)        =  6.98512E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.77883E-01  7.39500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.86953E+01  4.21938E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.38500E-02  1.38500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.60928E+00  5.69062E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.42032E+01  4.09864E+02 ];
CPU_USAGE                 (idx, 1)        = 8.75354 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98939E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.62911E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  5.76742E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60350E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.64413E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.47126E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.03605E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26557E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46922E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.45700E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65445E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.39372E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.75904E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06328E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17854E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44263E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35710E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17633E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55525E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.61048E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38179E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39185E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51294E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51894E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.28736E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.17043E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.07945E-04 -2.38338E+25  2.20820E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.81374E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.54352E+18 0.00391  1.64634E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  9.21844E+19 0.00047  9.83232E-01 6.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.57869E+14 0.25209  3.83071E-06 0.25209 ];
TH232_CAPT                (idx, [1:   4]) = [  9.29876E+19 0.00053  7.53521E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33801E+19 0.00127  1.08428E-01 0.00126 ];
U235_CAPT                 (idx, [1:   4]) = [  1.91067E+14 0.34940  1.54325E-06 0.34940 ];
SM149_CAPT                (idx, [1:   4]) = [  1.26889E+16 0.04371  1.02788E-04 0.04370 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001539 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14326E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001539 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5157773 5.16305E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3918817 3.92265E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 924949 9.25734E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001539 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.56348E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36466E+20 3.7E-06  2.36466E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37762E+19 1.2E-07  9.37762E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23443E+20 0.00031  1.10365E+20 0.00021  1.30782E+19 0.00197 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17219E+20 0.00018  2.04141E+20 0.00012  1.30782E+19 0.00197 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.39014E+20 0.00034  2.39014E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.13943E+22 0.00025  2.97970E+22 0.00022  1.59731E+21 0.00189 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.21270E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39346E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05925E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51195E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51195E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.74028E-01 0.07664 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.95507E-01 0.00896 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.56149E-04 0.01618 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.13019E+04 0.02387 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07433E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 9.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.90624E-01 0.06394 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.45249E-01 0.06394 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52160E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.89359E-01 0.00038  1.09551E-01 0.00038  3.56748E-04 0.00880 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.89119E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.89374E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.89119E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09000E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.99794E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  6.99818E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.37098E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.37047E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51520E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50900E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30062E-03 0.00509  2.64995E-04 0.02014  8.02706E-04 0.01123  6.14491E-04 0.01132  1.29281E-03 0.00828  2.73593E-04 0.01977  5.20313E-05 0.04605 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47120E-01 0.01618  1.24755E-02 3.0E-05  3.23380E-02 9.3E-05  1.06057E-01 0.00059  2.97331E-01 0.00026  1.23546E+00 0.00023  4.97795E+00 0.04070 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26157E-03 0.00776  2.62090E-04 0.02887  7.96566E-04 0.01575  6.09532E-04 0.01704  1.27846E-03 0.01290  2.67848E-04 0.03016  4.70738E-05 0.06984 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38323E-01 0.02341  1.24752E-02 4.3E-05  3.23412E-02 0.00014  1.06074E-01 0.00080  2.97326E-01 0.00037  1.23627E+00 0.00028  6.17393E+00 0.03081 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.10364E-07 0.00112  8.10013E-07 0.00112  9.17794E-07 0.02622 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.01702E-07 0.00104  8.01356E-07 0.00104  9.08034E-07 0.02622 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.25092E-03 0.00896  2.54699E-04 0.03061  7.97450E-04 0.01813  6.08662E-04 0.01954  1.27763E-03 0.01346  2.62612E-04 0.03146  4.98728E-05 0.07197 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.41846E-01 0.02428  1.24762E-02 5.0E-05  3.23319E-02 0.00014  1.05976E-01 0.00093  2.97244E-01 0.00043  1.23585E+00 0.00039  6.29826E+00 0.04231 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.22876E-07 0.02000  7.22810E-07 0.02001  7.41360E-07 0.04996 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.15365E-07 0.02002  7.15300E-07 0.02002  7.33647E-07 0.04992 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85059E-03 0.04293  2.12759E-04 0.13266  7.43931E-04 0.07564  5.14975E-04 0.08902  1.10840E-03 0.06290  2.18076E-04 0.13962  5.24518E-05 0.27744 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.62282E-01 0.10696  1.24785E-02 7.1E-05  3.23335E-02 0.00048  1.05948E-01 0.00304  2.97551E-01 0.00170  1.23576E+00 0.00115  6.85016E+00 0.11375 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.84973E-03 0.04204  2.10840E-04 0.12965  7.45161E-04 0.07305  5.21073E-04 0.08622  1.10294E-03 0.06133  2.15212E-04 0.13866  5.45047E-05 0.27048 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.67540E-01 0.10830  1.24785E-02 7.1E-05  3.23309E-02 0.00048  1.05889E-01 0.00292  2.97545E-01 0.00170  1.23573E+00 0.00115  6.90915E+00 0.11212 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96085E+03 0.03872 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.04332E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.95733E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28896E-03 0.00538 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.08928E+03 0.00537 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.98123E-09 0.00107 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.07543E-05 0.00504  7.07472E-05 0.00504  1.29947E-06 0.45448 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.85630E-05 0.01824  7.84922E-05 0.01831  2.09662E-06 0.54372 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.47493E-04 0.01547  3.48138E-04 0.01545  1.39623E-04 0.44439 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68722E+01 0.01159 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.42694E+01 0.00027  3.99495E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 06:56:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01464E+00  1.00230E+00  1.00553E+00  9.94447E-01  1.00441E+00  1.01161E+00  1.00267E+00  9.99827E-01  1.00513E+00  1.01165E+00  1.00182E+00  9.94393E-01  1.00767E+00  9.98617E-01  1.00999E+00  9.98048E-01  1.00056E+00  9.36688E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.56608E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74339E-01 8.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40386E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54140E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59263E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.44401E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.43477E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.08791E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.05565E-01 0.00058  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333667 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33389E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33389E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.00409E+03 ;
RUNNING_TIME              (idx, 1)        =  1.14944E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.52267E-01  7.43833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.13652E+02  4.49569E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.68500E-02  5.30000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.49425E+01  6.33317E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.08651E+02  6.16505E+02 ];
CPU_USAGE                 (idx, 1)        = 8.73549 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98992E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.61170E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.38377E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65369E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.21888E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85555E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.29639E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49729E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49305E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.75820E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.39831E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.86107E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.97213E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89691E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60109E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49451E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34596E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33417E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54344E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.75145E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17931E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.00710E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51024E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.32428E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.50496E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.26581E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45044E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.75051E-04 -1.26969E+26  2.20923E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.97433E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.56545E+18 0.00406  1.66797E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  9.21605E+19 0.00051  9.81979E-01 7.1E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.02267E+16 0.05117  1.08917E-04 0.05115 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44381E+19 0.00052  7.48338E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34079E+19 0.00132  1.06249E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  3.55944E+15 0.08633  2.82377E-05 0.08645 ];
SM149_CAPT                (idx, [1:   4]) = [  5.87835E+16 0.01946  4.65899E-04 0.01950 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001665 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13952E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001665 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5205546 5.21061E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3871300 3.87510E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 924819 9.25677E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001665 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.00583E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36454E+20 3.7E-06  2.36454E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37757E+19 1.2E-07  9.37757E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26144E+20 0.00030  1.12942E+20 0.00021  1.32018E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19919E+20 0.00017  2.06718E+20 0.00012  1.32018E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42194E+20 0.00036  2.42194E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.18382E+22 0.00025  3.02309E+22 0.00023  1.60729E+21 0.00183 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24202E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42340E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07522E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51425E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51425E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.91062E-01 0.05851 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.00237E-01 0.01311 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.27672E-04 0.02109 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.02516E+04 0.02462 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07435E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.35684E-01 0.03937 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.67699E-01 0.03937 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52149E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99674E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77130E-01 0.00044  1.08216E-01 0.00043  3.47662E-04 0.00923 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.76845E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.76338E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.76845E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07655E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00473E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00278E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36172E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36421E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51004E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51595E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.31251E-03 0.00614  2.64136E-04 0.01842  8.23924E-04 0.01251  6.02727E-04 0.01274  1.29347E-03 0.00890  2.76759E-04 0.01914  5.14948E-05 0.04144 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45639E-01 0.01367  1.24758E-02 3.0E-05  3.23378E-02 9.0E-05  1.06106E-01 0.00060  2.97389E-01 0.00025  1.23559E+00 0.00024  5.19877E+00 0.03804 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24094E-03 0.00824  2.56169E-04 0.02829  8.11552E-04 0.01683  5.71479E-04 0.01981  1.28011E-03 0.01359  2.67756E-04 0.02723  5.38699E-05 0.06573 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.52122E-01 0.02159  1.24756E-02 4.2E-05  3.23440E-02 0.00014  1.06081E-01 0.00086  2.97336E-01 0.00036  1.23619E+00 0.00030  6.39017E+00 0.02970 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.22004E-07 0.00132  8.21661E-07 0.00132  9.28828E-07 0.02088 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.03154E-07 0.00123  8.02817E-07 0.00123  9.07865E-07 0.02100 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19148E-03 0.00931  2.68018E-04 0.03096  7.86944E-04 0.02030  5.80807E-04 0.02113  1.24001E-03 0.01459  2.66408E-04 0.03141  4.92977E-05 0.06961 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.37750E-01 0.02372  1.24750E-02 6.3E-05  3.23375E-02 0.00015  1.06039E-01 0.00102  2.97452E-01 0.00045  1.23636E+00 0.00037  5.78792E+00 0.04461 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.27147E-07 0.01973  7.26910E-07 0.01973  8.00602E-07 0.05647 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.10848E-07 0.01971  7.10617E-07 0.01972  7.82565E-07 0.05641 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.58312E-03 0.04287  2.39054E-04 0.11951  5.60983E-04 0.08232  4.75846E-04 0.08792  1.02274E-03 0.07134  2.27473E-04 0.13600  5.70238E-05 0.28569 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.82805E-01 0.12062  1.24762E-02 0.00013  3.23106E-02 0.00041  1.06313E-01 0.00342  2.96092E-01 0.00117  1.23530E+00 0.00128  5.84435E+00 0.13488 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.59604E-03 0.04141  2.34971E-04 0.11723  5.63183E-04 0.08067  4.94752E-04 0.08577  1.02566E-03 0.06875  2.22733E-04 0.13328  5.47412E-05 0.27665 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.75103E-01 0.11939  1.24762E-02 0.00013  3.23101E-02 0.00041  1.06331E-01 0.00343  2.96245E-01 0.00121  1.23510E+00 0.00129  5.84435E+00 0.13488 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.57320E+03 0.03819 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.16750E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.98022E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.13855E-03 0.00485 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84344E+03 0.00491 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.81620E-09 0.00088 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92689E-05 0.00622  6.92479E-05 0.00622  2.03055E-06 0.35957 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.63462E-05 0.02519  6.63965E-05 0.02516  1.11254E-06 0.55212 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.23592E-04 0.02007  2.23532E-04 0.02012  2.28741E-04 0.35102 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71292E+01 0.01124 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.43477E+01 0.00026  4.00900E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 07:42:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02120E+00  1.00685E+00  9.99995E-01  9.96535E-01  1.00441E+00  1.00388E+00  1.00128E+00  1.00131E+00  1.01030E+00  9.97540E-01  1.00654E+00  1.00243E+00  1.00178E+00  1.00458E+00  9.97020E-01  1.01356E+00  9.97745E-01  9.33037E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.55918E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74408E-01 8.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40561E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54278E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58655E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45597E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44673E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.10486E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.04664E-01 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333648 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33410E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33410E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.40070E+03 ;
RUNNING_TIME              (idx, 1)        =  1.60626E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.33950E-01  8.16833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.59227E+02  4.55748E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  8.12000E-02  1.43500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.15250E+01  6.58243E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.54087E+02  6.49669E+02 ];
CPU_USAGE                 (idx, 1)        = 8.72026 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98992E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.59351E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.50029E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66924E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.07621E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89325E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32183E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57579E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50574E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.81143E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.55954E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.01592E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.16711E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79443E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74281E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48810E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36194E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34644E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55994E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.40859E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42582E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07971E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53787E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.10933E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.66580E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.32440E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12634E-03 -2.48691E+26  2.21045E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.10748E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.58314E+18 0.00382  1.68764E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  9.19711E+19 0.00049  9.80445E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.87653E+16 0.02738  4.13352E-04 0.02739 ];
TH232_CAPT                (idx, [1:   4]) = [  9.53982E+19 0.00055  7.45003E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34082E+19 0.00132  1.04713E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  1.44525E+16 0.04234  1.12804E-04 0.04223 ];
SM149_CAPT                (idx, [1:   4]) = [  9.41206E+16 0.01547  7.35035E-04 0.01546 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002310 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16954E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002310 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5239736 5.24483E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3838709 3.84221E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 923865 9.24657E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002310 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.26660E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36448E+20 3.5E-06  2.36448E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37750E+19 1.1E-07  9.37750E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.27954E+20 0.00028  1.14592E+20 0.00021  1.33621E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.21729E+20 0.00016  2.08367E+20 0.00011  1.33621E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.44147E+20 0.00036  2.44147E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.21363E+22 0.00025  3.05095E+22 0.00023  1.62676E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.25758E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44305E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08685E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51411E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51411E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.10391E-01 0.05380 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19900E-01 0.01297 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.78651E-04 0.02354 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.61872E+03 0.02684 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07537E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.11933E-01 0.04090 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.46155E-01 0.04090 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52144E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.68826E-01 0.00041  1.07299E-01 0.00041  3.45007E-04 0.00920 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.68978E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.68505E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.68978E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06776E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00659E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00802E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35922E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35708E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53103E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52679E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.35350E-03 0.00567  2.70892E-04 0.01957  8.23823E-04 0.01066  6.06052E-04 0.01319  1.32679E-03 0.00906  2.75574E-04 0.01864  5.03640E-05 0.04625 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.44528E-01 0.01559  1.24760E-02 2.8E-05  3.23339E-02 9.4E-05  1.06108E-01 0.00057  2.97337E-01 0.00028  1.23560E+00 0.00025  5.07155E+00 0.04003 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25668E-03 0.00819  2.63327E-04 0.02851  8.05154E-04 0.01606  5.98920E-04 0.01958  1.27916E-03 0.01225  2.59611E-04 0.02958  5.05043E-05 0.06472 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44029E-01 0.02178  1.24757E-02 4.1E-05  3.23357E-02 0.00014  1.06138E-01 0.00083  2.97217E-01 0.00037  1.23617E+00 0.00031  6.35016E+00 0.02977 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.32196E-07 0.00121  8.31841E-07 0.00121  9.48975E-07 0.02364 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.06217E-07 0.00115  8.05872E-07 0.00116  9.19474E-07 0.02378 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20667E-03 0.00921  2.65519E-04 0.03159  7.94085E-04 0.01781  5.72641E-04 0.02306  1.26972E-03 0.01351  2.57280E-04 0.03154  4.74309E-05 0.07644 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.41790E-01 0.02636  1.24760E-02 4.9E-05  3.23304E-02 0.00015  1.06021E-01 0.00100  2.97374E-01 0.00048  1.23516E+00 0.00048  6.36582E+00 0.04190 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.35350E-07 0.01960  7.34859E-07 0.01960  9.62512E-07 0.16683 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.12578E-07 0.01960  7.12102E-07 0.01960  9.32204E-07 0.16614 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.77044E-03 0.04562  1.89746E-04 0.14560  7.04103E-04 0.08421  5.26516E-04 0.09426  1.10305E-03 0.06129  2.11374E-04 0.14312  3.56519E-05 0.33516 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.07882E-01 0.08967  1.24773E-02 0.00011  3.23535E-02 0.00059  1.06175E-01 0.00345  2.97640E-01 0.00173  1.23517E+00 0.00114  5.42330E+00 0.17033 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.76273E-03 0.04548  1.81584E-04 0.14265  7.18960E-04 0.08364  5.32513E-04 0.09382  1.08548E-03 0.06083  2.09430E-04 0.14144  3.47621E-05 0.32491 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.02399E-01 0.08506  1.24773E-02 0.00011  3.23537E-02 0.00058  1.06163E-01 0.00343  2.97574E-01 0.00168  1.23530E+00 0.00113  5.42330E+00 0.17033 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.78793E+03 0.04139 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.26722E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.00906E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.16272E-03 0.00589 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82635E+03 0.00595 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76945E-09 0.00084 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.78925E-05 0.00778  6.79081E-05 0.00783  1.59439E-06 0.38434 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.32899E-05 0.02948  6.34175E-05 0.02949  9.18329E-07 0.53413 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.76232E-04 0.02307  1.76120E-04 0.02311  2.06440E-04 0.37504 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72204E+01 0.01127 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44673E+01 0.00025  4.02020E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 08:29:31 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01781E+00  1.00699E+00  1.01128E+00  9.94238E-01  1.00152E+00  1.00287E+00  1.00570E+00  1.00355E+00  1.00932E+00  1.00386E+00  9.99828E-01  1.00540E+00  9.97707E-01  1.00223E+00  9.99030E-01  1.00413E+00  1.00750E+00  9.27035E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.55127E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74487E-01 8.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40571E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54251E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58402E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47323E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46400E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.13730E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.04060E-01 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333972 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33394E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33394E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.81250E+03 ;
RUNNING_TIME              (idx, 1)        =  2.08038E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.14167E-01  8.02167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.06531E+02  4.73039E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.44833E-02  1.32833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.83411E+01  6.81608E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.01265E+02  6.57948E+02 ];
CPU_USAGE                 (idx, 1)        = 8.71234 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99008E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.58380E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.52540E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66366E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.44530E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91432E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33497E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57956E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49861E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.98618E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.67927E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.28265E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.26255E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69791E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85290E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93626E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32549E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31849E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52148E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.37017E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81468E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.07582E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51020E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.48670E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70930E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.42249E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.28768E-03 -5.05111E+26  2.21301E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.36171E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58435E+18 0.00414  1.69042E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  9.15701E+19 0.00051  9.77073E-01 7.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.54078E+17 0.01211  1.64412E-03 0.01211 ];
TH232_CAPT                (idx, [1:   4]) = [  9.71480E+19 0.00051  7.40964E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33210E+19 0.00136  1.01602E-01 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  5.33651E+16 0.02235  4.06908E-04 0.02232 ];
SM149_CAPT                (idx, [1:   4]) = [  1.33918E+17 0.01369  1.02135E-03 0.01367 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001818 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15668E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001818 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5293897 5.29922E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3784298 3.78793E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 923623 9.24414E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001818 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.29105E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36435E+20 3.8E-06  2.36435E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37730E+19 1.2E-07  9.37730E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.30966E+20 0.00027  1.17338E+20 0.00020  1.36273E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.24739E+20 0.00016  2.11111E+20 0.00011  1.36273E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47416E+20 0.00036  2.47416E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.26476E+22 0.00026  3.09942E+22 0.00023  1.65336E+21 0.00197 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.28725E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47611E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10568E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51098E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51098E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.07471E-01 0.05626 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21667E-01 0.01448 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.52298E-04 0.02749 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.51001E+03 0.02831 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07561E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.38457E-01 0.04657 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.79414E-01 0.04657 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52135E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.55152E-01 0.00042  1.05780E-01 0.00042  3.42941E-04 0.00860 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55971E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.55651E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55971E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05338E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00728E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00910E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35827E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35560E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.55359E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54747E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44220E-03 0.00535  2.73743E-04 0.01884  8.36780E-04 0.01122  6.26756E-04 0.01318  1.36271E-03 0.00881  2.87457E-04 0.01862  5.47622E-05 0.04425 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53292E-01 0.01483  1.24761E-02 2.7E-05  3.23311E-02 1.0E-04  1.06109E-01 0.00058  2.97658E-01 0.00027  1.23673E+00 0.00030  5.30431E+00 0.03833 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26994E-03 0.00813  2.67530E-04 0.02850  7.80380E-04 0.01588  5.98604E-04 0.01919  1.29784E-03 0.01339  2.75145E-04 0.02978  5.04417E-05 0.06325 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.50553E-01 0.02173  1.24755E-02 4.2E-05  3.23336E-02 0.00014  1.06146E-01 0.00088  2.97702E-01 0.00038  1.23671E+00 0.00037  6.52264E+00 0.02871 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.45404E-07 0.00121  8.45152E-07 0.00121  9.24823E-07 0.01825 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.07454E-07 0.00116  8.07213E-07 0.00117  8.83301E-07 0.01825 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24440E-03 0.00887  2.49848E-04 0.03200  7.86839E-04 0.01797  6.03445E-04 0.02025  1.26994E-03 0.01552  2.74714E-04 0.03140  5.96131E-05 0.06383 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.72420E-01 0.02475  1.24770E-02 3.7E-05  3.23304E-02 0.00016  1.05984E-01 0.00092  2.97880E-01 0.00048  1.23711E+00 0.00057  6.61441E+00 0.03766 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.47534E-07 0.01958  7.47372E-07 0.01958  7.77491E-07 0.05408 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.14151E-07 0.01958  7.13996E-07 0.01958  7.43017E-07 0.05410 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91571E-03 0.04250  2.07813E-04 0.14901  7.20507E-04 0.08277  6.01143E-04 0.08654  1.13855E-03 0.06797  1.88919E-04 0.13840  5.87815E-05 0.25635 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.67538E-01 0.10356  1.24795E-02 5.1E-06  3.23231E-02 0.00059  1.05828E-01 0.00260  2.98010E-01 0.00180  1.23587E+00 0.00133  6.46791E+00 0.11030 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90210E-03 0.04169  2.03303E-04 0.14742  7.12537E-04 0.08194  5.85240E-04 0.08652  1.15206E-03 0.06725  1.88608E-04 0.13840  6.03469E-05 0.25318 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.72620E-01 0.10253  1.24795E-02 5.1E-06  3.23213E-02 0.00059  1.05823E-01 0.00259  2.98037E-01 0.00180  1.23588E+00 0.00133  6.46791E+00 0.11030 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.91748E+03 0.03791 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.38470E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.00822E-07 0.00045 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20177E-03 0.00602 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81912E+03 0.00607 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.75182E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69143E-05 0.01023  6.69248E-05 0.01023  1.57904E-06 0.38900 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.82658E-05 0.02881  5.81537E-05 0.02896  2.09460E-06 0.51784 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.50014E-04 0.02542  1.49829E-04 0.02527  2.14378E-04 0.37508 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74511E+01 0.01087 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46400E+01 0.00025  4.02891E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 09:17:16 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01693E+00  1.00780E+00  9.99449E-01  1.00265E+00  1.00393E+00  9.96460E-01  1.00205E+00  1.00754E+00  9.99376E-01  1.00807E+00  9.98067E-01  1.00778E+00  9.98327E-01  1.00304E+00  1.00312E+00  1.00462E+00  1.00430E+00  9.36487E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.55111E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74489E-01 8.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40170E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53859E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58280E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48221E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47293E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16755E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.06044E-01 0.00057  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333486 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33369E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33369E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.22669E+03 ;
RUNNING_TIME              (idx, 1)        =  2.55790E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.94150E-01  7.99833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.54168E+02  4.76371E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.09017E-01  1.45333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.52617E+01  6.92053E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.48912E+02  6.75497E+02 ];
CPU_USAGE                 (idx, 1)        = 8.70516 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99024E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.57527E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.59417E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67326E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.79934E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95021E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35875E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61201E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50542E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09560E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.78995E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.68792E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.42704E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25350E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94696E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35122E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33573E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32602E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53168E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.04860E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17308E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11579E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54186E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.28602E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.81229E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.50134E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85969E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.43225E-03 -7.57827E+26  2.21554E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.53636E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59788E+18 0.00373  1.70276E-02 0.00369 ];
U233_FISS                 (idx, [1:   4]) = [  9.13037E+19 0.00050  9.72983E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.26678E+17 0.00939  3.48137E-03 0.00940 ];
TH232_CAPT                (idx, [1:   4]) = [  9.83537E+19 0.00053  7.38058E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33041E+19 0.00132  9.98373E-02 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  1.16696E+17 0.01349  8.75600E-04 0.01346 ];
SM149_CAPT                (idx, [1:   4]) = [  1.48199E+17 0.01349  1.11218E-03 0.01349 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001067 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15179E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001067 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5323629 5.32948E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3749159 3.75292E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 928279 9.29114E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001067 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.08616E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36420E+20 3.7E-06  2.36420E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37702E+19 1.2E-07  9.37702E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.33323E+20 0.00031  1.19570E+20 0.00022  1.37530E+19 0.00205 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27093E+20 0.00018  2.13340E+20 0.00013  1.37530E+19 0.00205 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.50045E+20 0.00035  2.50045E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.30620E+22 0.00027  3.14021E+22 0.00024  1.65995E+21 0.00189 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.32325E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.50326E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11966E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50656E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50656E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.19714E-01 0.05370 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.38481E-01 0.01347 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.59915E-04 0.02588 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.68603E+03 0.02526 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07091E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.37112E-01 0.03726 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.68459E-01 0.03726 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52126E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.45955E-01 0.00043  1.04791E-01 0.00043  3.44201E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.45543E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.45544E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.45543E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04239E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00718E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00813E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35840E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35691E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56914E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.56955E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49469E-03 0.00560  2.75516E-04 0.01960  8.52784E-04 0.01069  6.50908E-04 0.01270  1.38167E-03 0.00896  2.84749E-04 0.01932  4.90634E-05 0.04677 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.39281E-01 0.01355  1.24757E-02 3.0E-05  3.23230E-02 9.0E-05  1.06208E-01 0.00061  2.97685E-01 0.00026  1.23663E+00 0.00030  5.28179E+00 0.04084 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28194E-03 0.00782  2.61709E-04 0.02817  8.00589E-04 0.01728  6.12324E-04 0.01765  1.29663E-03 0.01262  2.66151E-04 0.02761  4.45344E-05 0.06793 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38354E-01 0.02251  1.24756E-02 4.1E-05  3.23270E-02 0.00015  1.06245E-01 0.00080  2.97519E-01 0.00036  1.23638E+00 0.00042  6.66197E+00 0.02978 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.56194E-07 0.00128  8.55900E-07 0.00128  9.43224E-07 0.01592 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.09865E-07 0.00116  8.09587E-07 0.00116  8.92017E-07 0.01586 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27292E-03 0.00917  2.49648E-04 0.03244  7.94652E-04 0.01940  6.15247E-04 0.02223  1.30492E-03 0.01492  2.64117E-04 0.03162  4.43343E-05 0.07906 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.36044E-01 0.02485  1.24757E-02 4.9E-05  3.23199E-02 0.00016  1.06265E-01 0.00100  2.97476E-01 0.00045  1.23735E+00 0.00050  6.49278E+00 0.04390 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.57892E-07 0.01962  7.57796E-07 0.01962  7.84131E-07 0.05003 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.17095E-07 0.01962  7.17005E-07 0.01962  7.41990E-07 0.05000 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.75531E-03 0.04358  1.94873E-04 0.16819  8.22950E-04 0.08025  4.57002E-04 0.09064  1.05137E-03 0.06753  1.93002E-04 0.15221  3.61090E-05 0.38514 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.84525E-01 0.09876  1.24770E-02 0.00014  3.23035E-02 0.00041  1.05826E-01 0.00275  2.97407E-01 0.00170  1.23496E+00 0.00146  5.42330E+00 0.17033 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.76345E-03 0.04272  1.98897E-04 0.16269  8.14644E-04 0.08024  4.67934E-04 0.09083  1.04634E-03 0.06505  1.97842E-04 0.15510  3.77914E-05 0.37066 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.89032E-01 0.10057  1.24770E-02 0.00014  3.23036E-02 0.00042  1.05893E-01 0.00284  2.97393E-01 0.00170  1.23494E+00 0.00146  5.42330E+00 0.17033 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.66084E+03 0.03944 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.49629E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.03664E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20829E-03 0.00634 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77613E+03 0.00631 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77614E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.57908E-05 0.00881  6.57621E-05 0.00889  1.47234E-06 0.39236 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88100E-05 0.03006  5.88980E-05 0.03010  1.17877E-06 0.46491 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.57752E-04 0.02462  1.57601E-04 0.02438  2.03050E-04 0.37516 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69364E+01 0.01064 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47293E+01 0.00025  4.03227E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 10:06:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01939E+00  1.00309E+00  9.98848E-01  1.00052E+00  1.00307E+00  1.00207E+00  1.00884E+00  1.00460E+00  1.00120E+00  1.00956E+00  1.00485E+00  1.00419E+00  1.00083E+00  9.98647E-01  1.00295E+00  1.00304E+00  1.00207E+00  9.32242E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.55199E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74480E-01 8.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39770E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53472E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58166E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49909E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48983E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.21231E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.09054E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334542 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33392E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33392E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.65085E+03 ;
RUNNING_TIME              (idx, 1)        =  3.04677E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.76850E-01  8.27000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.02935E+02  4.87674E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.22867E-01  1.38500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.23323E+01  7.07050E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.97650E+02  6.79519E+02 ];
CPU_USAGE                 (idx, 1)        = 8.70051 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99035E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.56970E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.62155E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67077E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.80852E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97873E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37599E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61051E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50087E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.26583E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.92328E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.67788E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.58522E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93536E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06387E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06818E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30681E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30296E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50046E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.87290E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78573E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.12052E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54773E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.78727E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85923E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.60972E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43282E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.81538E-03 -1.28401E+27  2.22080E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.85799E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.62198E+18 0.00412  1.73111E-02 0.00403 ];
U233_FISS                 (idx, [1:   4]) = [  9.03150E+19 0.00052  9.63986E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  7.89470E+17 0.00598  8.42635E-03 0.00595 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00175E+20 0.00053  7.32546E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31879E+19 0.00141  9.64396E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  2.76263E+17 0.00954  2.02020E-03 0.00952 ];
SM149_CAPT                (idx, [1:   4]) = [  1.62879E+17 0.01212  1.19143E-03 0.01220 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001749 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14988E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001749 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5385691 5.39110E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3690020 3.69356E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 926038 9.26843E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001749 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.45755E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36384E+20 3.7E-06  2.36384E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37630E+19 1.1E-07  9.37630E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36716E+20 0.00027  1.22708E+20 0.00020  1.40083E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30479E+20 0.00016  2.16471E+20 0.00011  1.40083E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.53657E+20 0.00037  2.53657E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36572E+22 0.00025  3.19655E+22 0.00023  1.69171E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35106E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.53990E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14012E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49777E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49777E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.69296E-01 0.05355 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.32243E-01 0.01342 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.53499E-04 0.02470 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.01338E+03 0.02862 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07318E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.94138E-01 0.03998 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.29755E-01 0.03998 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52108E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.31125E-01 0.00044  1.03128E-01 0.00042  3.36294E-04 0.00976 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.31766E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.31942E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.31766E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02693E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00538E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00530E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36083E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36078E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.61205E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.61114E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.55110E-03 0.00618  2.91606E-04 0.02036  8.55021E-04 0.01189  6.51779E-04 0.01309  1.38866E-03 0.00894  3.05462E-04 0.01730  5.85777E-05 0.04235 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.52920E-01 0.01373  1.24754E-02 3.0E-05  3.23203E-02 0.00010  1.06142E-01 0.00055  2.98029E-01 0.00028  1.23960E+00 0.00041  5.21685E+00 0.03617 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28871E-03 0.00833  2.65118E-04 0.03020  8.07829E-04 0.01706  5.95566E-04 0.01854  1.29058E-03 0.01379  2.78042E-04 0.02718  5.15780E-05 0.05862 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.47895E-01 0.01956  1.24754E-02 3.9E-05  3.23160E-02 0.00013  1.06089E-01 0.00078  2.98125E-01 0.00039  1.23998E+00 0.00056  6.34561E+00 0.02766 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.74608E-07 0.00124  8.74259E-07 0.00124  9.85220E-07 0.02010 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14321E-07 0.00116  8.13996E-07 0.00116  9.17497E-07 0.02021 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24806E-03 0.00994  2.70650E-04 0.03360  7.73742E-04 0.01886  5.97083E-04 0.02239  1.27319E-03 0.01510  2.77871E-04 0.02940  5.55266E-05 0.07023 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.57623E-01 0.02446  1.24758E-02 5.3E-05  3.23250E-02 0.00018  1.05994E-01 0.00096  2.97894E-01 0.00043  1.23940E+00 0.00075  6.28757E+00 0.04097 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.73018E-07 0.01961  7.72684E-07 0.01961  8.56682E-07 0.05443 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20047E-07 0.01961  7.19735E-07 0.01961  7.98259E-07 0.05450 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.68598E-03 0.04317  1.90297E-04 0.15217  6.31152E-04 0.08386  5.06037E-04 0.09084  1.08619E-03 0.06215  2.34053E-04 0.13028  3.82534E-05 0.33406 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42875E-01 0.08103  1.24731E-02 0.00020  3.23552E-02 0.00074  1.06515E-01 0.00358  2.97832E-01 0.00165  1.23511E+00 0.00140  6.32329E+00 0.14044 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.68363E-03 0.04326  1.91574E-04 0.14946  6.31663E-04 0.08328  5.10075E-04 0.09059  1.07715E-03 0.06159  2.32540E-04 0.13007  4.06369E-05 0.33760 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.44852E-01 0.08159  1.24733E-02 0.00020  3.23552E-02 0.00074  1.06498E-01 0.00354  2.97877E-01 0.00163  1.23519E+00 0.00142  6.32329E+00 0.14044 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.48931E+03 0.03876 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.69293E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09376E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17543E-03 0.00578 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.65302E+03 0.00575 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.79532E-09 0.00068 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.66635E-05 0.00835  6.66729E-05 0.00836  3.91738E-07 0.70818 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.04055E-05 0.02862  6.04143E-05 0.02860  2.63474E-07 0.86906 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.51851E-04 0.02334  1.52191E-04 0.02332  4.91374E-05 0.70736 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68241E+01 0.01162 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48983E+01 0.00026  4.03820E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 10:55:49 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01228E+00  1.00407E+00  1.00433E+00  1.00572E+00  1.00366E+00  1.00147E+00  1.00241E+00  1.00317E+00  9.98430E-01  1.01229E+00  1.00585E+00  1.00235E+00  9.99009E-01  1.00075E+00  1.00828E+00  1.00419E+00  1.00831E+00  9.23429E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.55415E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74458E-01 7.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39349E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53070E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58148E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50165E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49236E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23142E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.11225E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333670 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33379E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33379E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.08153E+03 ;
RUNNING_TIME              (idx, 1)        =  3.54341E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.58917E-01  8.20667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.52479E+02  4.95434E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.35567E-01  1.27000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.95377E+01  7.20537E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.47177E+02  6.89212E+02 ];
CPU_USAGE                 (idx, 1)        = 8.69651 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99052E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.56496E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.69894E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68288E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.51166E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00999E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39236E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.65595E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51071E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.74467E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22289E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.06287E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.84686E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.22254E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.33503E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32879E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.30034E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29472E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.49088E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.59638E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27915E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.15660E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.64673E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.82499E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94647E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.74631E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28656E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.18809E-02 -2.62327E+27  2.23420E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02122E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.64775E+18 0.00396  1.75803E-02 0.00389 ];
U233_FISS                 (idx, [1:   4]) = [  8.81164E+19 0.00054  9.40185E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.25631E+18 0.00366  2.40734E-02 0.00358 ];
PU239_FISS                (idx, [1:   4]) = [  6.18342E+14 0.20455  6.59391E-06 0.20490 ];
PU240_FISS                (idx, [1:   4]) = [  2.57642E+13 1.00000  2.76167E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01237E+20 0.00052  7.19272E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28739E+19 0.00141  9.14667E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  7.94168E+17 0.00627  5.64248E-03 0.00626 ];
PU239_CAPT                (idx, [1:   4]) = [  3.59910E+14 0.26140  2.55876E-06 0.26140 ];
PU240_CAPT                (idx, [1:   4]) = [  1.03852E+14 0.49752  7.30394E-07 0.49752 ];
SM149_CAPT                (idx, [1:   4]) = [  1.68806E+17 0.01262  1.19936E-03 0.01262 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001370 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12016E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001370 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5445491 5.45102E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3626219 3.62972E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 929660 9.30458E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001370 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.62050E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36290E+20 4.0E-06  2.36290E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37416E+19 1.1E-07  9.37416E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40717E+20 0.00028  1.26453E+20 0.00021  1.42639E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34459E+20 0.00017  2.20195E+20 0.00012  1.42639E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58210E+20 0.00036  2.58210E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43054E+22 0.00025  3.25910E+22 0.00022  1.71438E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40263E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58485E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16122E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47573E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47573E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.09453E-01 0.05625 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23629E-01 0.01486 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.50518E-04 0.02577 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.06831E+03 0.02804 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06957E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.55684E-01 0.04244 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.94666E-01 0.04244 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52065E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99746E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.14836E-01 0.00046  1.01327E-01 0.00045  3.32714E-04 0.00890 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.15185E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.15140E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.15185E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00908E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.97626E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.97839E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.40114E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.39791E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.71736E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.70853E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.68829E-03 0.00559  2.93025E-04 0.01950  8.87430E-04 0.01069  6.64114E-04 0.01241  1.46015E-03 0.00892  3.21628E-04 0.01898  6.19412E-05 0.04324 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.60899E-01 0.01458  1.24756E-02 2.9E-05  3.22972E-02 0.00010  1.06262E-01 0.00054  2.98888E-01 0.00028  1.24543E+00 0.00054  5.41571E+00 0.03561 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.33219E-03 0.00778  2.66385E-04 0.02806  8.05169E-04 0.01633  6.02265E-04 0.01877  1.31374E-03 0.01267  2.85819E-04 0.03136  5.88083E-05 0.06019 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.63003E-01 0.02249  1.24753E-02 3.8E-05  3.22935E-02 0.00014  1.06122E-01 0.00074  2.98790E-01 0.00042  1.24475E+00 0.00065  6.29776E+00 0.02799 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.83411E-07 0.00126  8.83055E-07 0.00126  9.84819E-07 0.01633 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.08127E-07 0.00118  8.07801E-07 0.00118  9.00788E-07 0.01628 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28134E-03 0.00899  2.60288E-04 0.03309  8.02788E-04 0.01806  5.83144E-04 0.02227  1.29385E-03 0.01523  2.86465E-04 0.03121  5.48124E-05 0.06891 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.66549E-01 0.02370  1.24755E-02 5.1E-05  3.22873E-02 0.00019  1.06239E-01 0.00109  2.98761E-01 0.00049  1.24616E+00 0.00105  6.74535E+00 0.03728 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.82682E-07 0.01971  7.82400E-07 0.01971  8.36153E-07 0.05865 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.16190E-07 0.01970  7.15932E-07 0.01971  7.65149E-07 0.05869 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85885E-03 0.04258  2.29380E-04 0.13954  7.21528E-04 0.07653  4.97759E-04 0.09396  1.13155E-03 0.06379  2.29275E-04 0.11728  4.93585E-05 0.30719 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.97685E-01 0.10892  1.24748E-02 0.00017  3.22709E-02 0.00058  1.05777E-01 0.00277  2.99183E-01 0.00199  1.24741E+00 0.00350  7.18168E+00 0.11460 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.86802E-03 0.04197  2.40985E-04 0.13368  7.15907E-04 0.07623  5.03802E-04 0.09107  1.12036E-03 0.06263  2.36684E-04 0.11897  5.02827E-05 0.31093 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.90656E-01 0.10473  1.24747E-02 0.00017  3.22719E-02 0.00058  1.05770E-01 0.00276  2.99201E-01 0.00199  1.24747E+00 0.00349  7.18168E+00 0.11460 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.69296E+03 0.03865 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.76535E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.01834E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32220E-03 0.00541 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.79098E+03 0.00550 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.79836E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.58746E-05 0.00884  6.58336E-05 0.00886  1.80600E-06 0.35928 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.51084E-05 0.02811  5.49538E-05 0.02807  2.00003E-06 0.46030 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.47606E-04 0.02581  1.47347E-04 0.02584  2.10118E-04 0.35145 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68492E+01 0.01192 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49236E+01 0.00027  4.01713E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 11:46:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01521E+00  1.00442E+00  1.00718E+00  1.00319E+00  1.00797E+00  1.00070E+00  1.00553E+00  1.00005E+00  1.00521E+00  1.00118E+00  9.90753E-01  9.99716E-01  1.00153E+00  1.00490E+00  1.01330E+00  1.00649E+00  9.98108E-01  9.34565E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.56161E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74384E-01 9.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39303E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53058E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58154E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48777E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47844E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20632E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.12805E-01 0.00061  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333604 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33382E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33382E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.51717E+03 ;
RUNNING_TIME              (idx, 1)        =  4.04622E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.49700E-01  9.07833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.02625E+02  5.01459E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.49017E-01  1.34500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.68703E+01  7.33258E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.97337E+02  6.95222E+02 ];
CPU_USAGE                 (idx, 1)        = 8.69249 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99051E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.56032E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.87247E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.72854E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.46019E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02447E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39662E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.81418E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55520E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.08715E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77299E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.00694E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.22688E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04479E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84332E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38065E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37444E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34565E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56378E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88006E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30454E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26229E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.90406E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.43282E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10837E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.79855E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57630E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.25470E-02 -4.97829E+27  2.25775E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03282E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58980E+18 0.00385  1.69866E-02 0.00379 ];
U233_FISS                 (idx, [1:   4]) = [  8.46508E+19 0.00054  9.04502E-01 0.00016 ];
U235_FISS                 (idx, [1:   4]) = [  4.62695E+18 0.00235  4.94402E-02 0.00231 ];
U238_FISS                 (idx, [1:   4]) = [  5.13567E+13 0.70596  5.49403E-07 0.70601 ];
PU239_FISS                (idx, [1:   4]) = [  1.93807E+16 0.03689  2.07059E-04 0.03685 ];
PU240_FISS                (idx, [1:   4]) = [  4.17702E+14 0.24365  4.44765E-06 0.24366 ];
PU241_FISS                (idx, [1:   4]) = [  4.14466E+14 0.24367  4.43865E-06 0.24366 ];
TH232_CAPT                (idx, [1:   4]) = [  9.92062E+19 0.00051  6.96805E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23760E+19 0.00133  8.69279E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  1.63483E+18 0.00389  1.14830E-02 0.00389 ];
U238_CAPT                 (idx, [1:   4]) = [  7.85423E+13 0.57544  5.51000E-07 0.57547 ];
PU239_CAPT                (idx, [1:   4]) = [  8.72146E+15 0.05361  6.12715E-05 0.05358 ];
PU240_CAPT                (idx, [1:   4]) = [  1.76483E+15 0.13066  1.24000E-05 0.13064 ];
PU241_CAPT                (idx, [1:   4]) = [  7.88668E+13 0.57544  5.54176E-07 0.57544 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75163E+17 0.01219  1.23032E-03 0.01219 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001456 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13124E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001456 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5471304 5.47694E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3596830 3.60024E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 933322 9.34129E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001456 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.71946E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36163E+20 4.1E-06  2.36163E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37037E+19 1.1E-07  9.37037E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42431E+20 0.00028  1.28036E+20 0.00021  1.43954E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36135E+20 0.00017  2.21739E+20 0.00012  1.43954E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59952E+20 0.00034  2.59952E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44694E+22 0.00025  3.27415E+22 0.00022  1.72791E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.42837E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60419E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16544E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.41743E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.41743E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.71193E-01 0.05576 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.45644E-01 0.01240 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.52990E-04 0.02695 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.11334E+03 0.02797 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06589E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.34154E-01 0.04401 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.74912E-01 0.04401 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52032E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99827E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07336E-01 0.00044  1.00478E-01 0.00043  3.38414E-04 0.00871 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07886E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08520E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07886E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00136E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92261E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92213E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47834E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47879E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83785E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.84544E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.79967E-03 0.00564  2.92075E-04 0.01896  8.87041E-04 0.01052  6.97021E-04 0.01237  1.52703E-03 0.00869  3.25556E-04 0.01743  7.09435E-05 0.04252 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.83998E-01 0.01535  1.24762E-02 3.5E-05  3.22739E-02 0.00012  1.06463E-01 0.00057  2.99985E-01 0.00028  1.25307E+00 0.00072  6.02955E+00 0.03058 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.36966E-03 0.00780  2.59268E-04 0.02769  7.89920E-04 0.01567  6.12511E-04 0.01832  1.34470E-03 0.01222  2.95674E-04 0.02514  6.75908E-05 0.05399 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.97692E-01 0.02091  1.24764E-02 4.5E-05  3.22804E-02 0.00018  1.06450E-01 0.00076  2.99833E-01 0.00039  1.25349E+00 0.00099  6.96345E+00 0.02304 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.65991E-07 0.00121  8.65607E-07 0.00121  9.79962E-07 0.01635 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.85705E-07 0.00115  7.85356E-07 0.00115  8.89166E-07 0.01636 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.35070E-03 0.00897  2.51618E-04 0.03209  7.90084E-04 0.01687  6.05807E-04 0.02152  1.33353E-03 0.01411  3.02214E-04 0.03025  6.74465E-05 0.06798 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.93758E-01 0.02552  1.24772E-02 9.4E-05  3.22860E-02 0.00023  1.06395E-01 0.00095  2.99747E-01 0.00053  1.25156E+00 0.00105  6.63445E+00 0.03329 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63996E-07 0.01961  7.63869E-07 0.01962  7.99351E-07 0.05277 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.93220E-07 0.01960  6.93107E-07 0.01960  7.24745E-07 0.05257 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.79336E-03 0.04523  2.25316E-04 0.16352  6.40228E-04 0.08681  4.90428E-04 0.09387  1.14870E-03 0.06387  1.99209E-04 0.13499  8.94778E-05 0.27278 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.07316E-01 0.10029  1.24790E-02 8.8E-05  3.22561E-02 0.00073  1.06365E-01 0.00304  2.99940E-01 0.00196  1.25282E+00 0.00343  6.69279E+00 0.09473 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.80314E-03 0.04511  2.26767E-04 0.15761  6.40551E-04 0.08604  4.88565E-04 0.09304  1.15559E-03 0.06408  2.07415E-04 0.13412  8.42520E-05 0.26708 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.09114E-01 0.09848  1.24790E-02 8.8E-05  3.22562E-02 0.00072  1.06332E-01 0.00297  2.99842E-01 0.00193  1.25261E+00 0.00342  6.70648E+00 0.09418 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.66974E+03 0.04077 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.58707E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.79095E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.36408E-03 0.00575 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.91829E+03 0.00581 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78366E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.58320E-05 0.01161  6.58123E-05 0.01163  1.49937E-06 0.44153 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.68298E-05 0.02956  5.68663E-05 0.02959  1.12790E-06 0.50788 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.49510E-04 0.02569  1.49479E-04 0.02572  1.62047E-04 0.40593 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65571E+01 0.01106 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47844E+01 0.00025  3.96981E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 12:36:19 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02543E+00  1.00423E+00  1.00454E+00  1.00208E+00  1.00876E+00  9.99884E-01  9.98282E-01  1.00109E+00  1.00254E+00  9.99796E-01  9.92020E-01  1.00612E+00  9.99605E-01  9.99188E-01  1.00888E+00  1.00812E+00  1.00708E+00  9.32353E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.56386E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74361E-01 7.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39430E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53191E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58149E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48058E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47133E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18848E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.12095E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333959 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33410E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33410E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.95152E+03 ;
RUNNING_TIME              (idx, 1)        =  4.54847E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.37033E-01  8.73333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.52678E+02  5.00530E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.01433E-01  5.24167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.42379E+01  7.36757E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.47523E+02  6.98951E+02 ];
CPU_USAGE                 (idx, 1)        = 8.68757 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99046E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.55570E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.90904E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73807E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.38107E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02142E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39386E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85352E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56475E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.00736E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99060E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.83241E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.41705E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13443E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04091E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81723E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38413E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35160E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57268E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.93243E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73469E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28126E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96700E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.86595E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.13593E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.79929E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21958E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.76619E-02 -6.10766E+27  2.26904E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03071E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.57794E+18 0.00404  1.68470E-02 0.00403 ];
U233_FISS                 (idx, [1:   4]) = [  8.36446E+19 0.00054  8.93014E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  5.34034E+18 0.00223  5.70161E-02 0.00219 ];
U238_FISS                 (idx, [1:   4]) = [  2.61915E+13 1.00000  2.77963E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  4.33789E+16 0.02452  4.63198E-04 0.02451 ];
PU240_FISS                (idx, [1:   4]) = [  1.87256E+15 0.12661  1.99484E-05 0.12665 ];
PU241_FISS                (idx, [1:   4]) = [  1.71265E+15 0.11507  1.83027E-05 0.11514 ];
TH232_CAPT                (idx, [1:   4]) = [  9.80835E+19 0.00054  6.88208E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22370E+19 0.00140  8.58618E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.87221E+18 0.00372  1.31366E-02 0.00371 ];
U238_CAPT                 (idx, [1:   4]) = [  2.84274E+14 0.32384  2.00279E-06 0.32425 ];
PU239_CAPT                (idx, [1:   4]) = [  2.09924E+16 0.03676  1.47349E-04 0.03679 ];
PU240_CAPT                (idx, [1:   4]) = [  4.10609E+15 0.08228  2.88130E-05 0.08235 ];
PU241_CAPT                (idx, [1:   4]) = [  2.34331E+14 0.32887  1.64697E-06 0.32886 ];
SM149_CAPT                (idx, [1:   4]) = [  1.68675E+17 0.01292  1.18342E-03 0.01290 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002305 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11182E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002305 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5477071 5.48206E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3599787 3.60286E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 925447 9.26192E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002305 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.08033E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36149E+20 4.1E-06  2.36149E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36896E+19 1.0E-07  9.36896E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42495E+20 0.00031  1.28043E+20 0.00021  1.44521E+19 0.00200 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36185E+20 0.00019  2.21733E+20 0.00012  1.44521E+19 0.00200 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59976E+20 0.00032  2.59976E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44314E+22 0.00023  3.26955E+22 0.00021  1.73589E+21 0.00197 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40794E+19 0.00159 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60264E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16368E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.38629E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.38629E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.24190E-01 0.05661 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33228E-01 0.01482 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.47068E-04 0.02762 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.17674E+03 0.02738 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07382E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99999E-01 3.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.37325E-01 0.04369 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.78241E-01 0.04369 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52055E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99857E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07899E-01 0.00046  1.00557E-01 0.00045  3.43848E-04 0.00951 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08362E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08377E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08362E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00108E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.89994E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90045E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.51215E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.51118E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.89007E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.89300E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.82518E-03 0.00561  2.91503E-04 0.01964  8.84526E-04 0.01145  6.98811E-04 0.01303  1.52933E-03 0.00891  3.45056E-04 0.01785  7.59582E-05 0.03590 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.97267E-01 0.01416  1.24766E-02 2.7E-05  3.22564E-02 0.00012  1.06501E-01 0.00054  3.00474E-01 0.00031  1.25541E+00 0.00069  6.29479E+00 0.02650 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42329E-03 0.00771  2.56637E-04 0.02894  8.00364E-04 0.01589  6.29022E-04 0.01816  1.36442E-03 0.01247  3.06683E-04 0.02685  6.61655E-05 0.05340 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.97997E-01 0.02079  1.24764E-02 3.5E-05  3.22540E-02 0.00019  1.06481E-01 0.00077  3.00552E-01 0.00044  1.25465E+00 0.00099  6.92921E+00 0.02218 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.58601E-07 0.00119  8.58146E-07 0.00118  9.94976E-07 0.01904 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.79477E-07 0.00111  7.79065E-07 0.00111  9.03156E-07 0.01899 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40061E-03 0.00958  2.51113E-04 0.03436  7.89748E-04 0.02041  6.13046E-04 0.02173  1.36454E-03 0.01538  3.13803E-04 0.02749  6.83660E-05 0.06711 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.97596E-01 0.02596  1.24764E-02 5.3E-05  3.22536E-02 0.00021  1.06342E-01 0.00089  3.00643E-01 0.00055  1.25505E+00 0.00123  6.52513E+00 0.03443 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.60424E-07 0.01973  7.59854E-07 0.01972  9.22915E-07 0.06614 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.90670E-07 0.01972  6.90153E-07 0.01971  8.38030E-07 0.06607 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13647E-03 0.04391  1.82708E-04 0.13678  7.62635E-04 0.08506  6.20713E-04 0.08667  1.15806E-03 0.06279  3.37165E-04 0.12586  7.51824E-05 0.23668 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.54877E-01 0.10781  1.24753E-02 0.00016  3.22286E-02 0.00093  1.06595E-01 0.00321  2.99778E-01 0.00189  1.25610E+00 0.00361  6.76320E+00 0.09541 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.13724E-03 0.04336  1.81277E-04 0.13346  7.51960E-04 0.08336  6.22583E-04 0.08482  1.18262E-03 0.06188  3.26933E-04 0.12472  7.18690E-05 0.24979 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.41738E-01 0.10659  1.24753E-02 0.00016  3.22297E-02 0.00093  1.06569E-01 0.00317  2.99755E-01 0.00187  1.25599E+00 0.00359  6.75493E+00 0.09555 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.16676E+03 0.03977 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.51042E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.72607E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43788E-03 0.00583 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.04070E+03 0.00593 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76408E-09 0.00068 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73221E-05 0.01079  6.73281E-05 0.01079  6.38514E-07 0.57751 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.32230E-05 0.03063  5.33216E-05 0.03069  2.26172E-07 0.69670 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.45367E-04 0.02598  1.45627E-04 0.02598  8.48295E-05 0.57955 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65380E+01 0.01132 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47133E+01 0.00028  3.94896E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 13:27:33 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00976E+00  1.00451E+00  1.00026E+00  1.00418E+00  1.00618E+00  1.00744E+00  1.00763E+00  1.00786E+00  1.00741E+00  1.00010E+00  1.00756E+00  9.99435E-01  1.00483E+00  1.00698E+00  1.00061E+00  1.00046E+00  1.00444E+00  9.20379E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.56803E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74320E-01 8.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39316E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53100E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58480E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47176E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46247E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17556E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.12690E-01 0.00057  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333181 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33390E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33390E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.39486E+03 ;
RUNNING_TIME              (idx, 1)        =  5.06074E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.24600E-01  8.75667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.03767E+02  5.10893E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.13617E-01  1.21833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.17573E+01  7.51933E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.98597E+02  6.99208E+02 ];
CPU_USAGE                 (idx, 1)        = 8.68422 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99057E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.55178E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.91029E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73911E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.90867E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00678E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38356E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86937E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56678E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.06227E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17186E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.80706E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.58177E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21150E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20507E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20232E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37680E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34482E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56406E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95723E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11694E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27451E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99244E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.19476E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11681E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.80237E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86286E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.28225E-02 -7.24708E+27  2.28043E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02959E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.56862E+18 0.00419  1.67505E-02 0.00414 ];
U233_FISS                 (idx, [1:   4]) = [  8.28001E+19 0.00061  8.84191E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  5.84339E+18 0.00201  6.24003E-02 0.00195 ];
PU239_FISS                (idx, [1:   4]) = [  8.01999E+16 0.01781  8.56629E-04 0.01785 ];
PU240_FISS                (idx, [1:   4]) = [  3.89388E+15 0.07812  4.15840E-05 0.07811 ];
PU241_FISS                (idx, [1:   4]) = [  4.15279E+15 0.07725  4.43617E-05 0.07733 ];
TH232_CAPT                (idx, [1:   4]) = [  9.71378E+19 0.00051  6.81419E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20670E+19 0.00150  8.46502E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.06793E+18 0.00368  1.45068E-02 0.00368 ];
U238_CAPT                 (idx, [1:   4]) = [  3.86809E+14 0.26913  2.72428E-06 0.26947 ];
PU239_CAPT                (idx, [1:   4]) = [  3.85614E+16 0.02450  2.70503E-04 0.02450 ];
PU240_CAPT                (idx, [1:   4]) = [  9.52884E+15 0.05428  6.68371E-05 0.05433 ];
PU241_CAPT                (idx, [1:   4]) = [  1.01181E+15 0.14962  7.09864E-06 0.14962 ];
SM149_CAPT                (idx, [1:   4]) = [  1.76536E+17 0.01204  1.23852E-03 0.01206 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001689 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11408E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001689 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5475737 5.48115E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3597375 3.60064E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 928577 9.29352E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001689 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.60187E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36158E+20 4.3E-06  2.36158E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36771E+19 1.0E-07  9.36771E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42502E+20 0.00028  1.28071E+20 0.00019  1.44315E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36179E+20 0.00017  2.21748E+20 0.00011  1.44315E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60079E+20 0.00035  2.60079E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44087E+22 0.00025  3.26753E+22 0.00022  1.73336E+21 0.00192 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41710E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60350E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16182E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.35583E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.35583E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.85167E-01 0.05392 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23017E-01 0.01483 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.50637E-04 0.02585 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.12381E+03 0.02638 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07067E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.54324E-01 0.04213 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.93514E-01 0.04213 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52098E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99884E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07627E-01 0.00048  1.00511E-01 0.00046  3.45579E-04 0.00869 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08090E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08057E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08090E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00114E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88476E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88181E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53536E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53963E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.92783E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.93244E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.89496E-03 0.00533  2.95644E-04 0.01875  9.07823E-04 0.01047  6.96634E-04 0.01243  1.56051E-03 0.00813  3.60196E-04 0.01657  7.41507E-05 0.03742 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.94865E-01 0.01309  1.24759E-02 3.7E-05  3.22510E-02 0.00013  1.06752E-01 0.00055  3.00652E-01 0.00030  1.25777E+00 0.00077  6.30443E+00 0.02717 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44652E-03 0.00734  2.62200E-04 0.02949  7.95568E-04 0.01533  6.21971E-04 0.01711  1.38103E-03 0.01211  3.21519E-04 0.02297  6.42270E-05 0.05768 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.91484E-01 0.02059  1.24757E-02 5.5E-05  3.22476E-02 0.00019  1.06798E-01 0.00077  3.00565E-01 0.00043  1.25678E+00 0.00098  6.89207E+00 0.02224 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.51156E-07 0.00135  8.50759E-07 0.00136  9.65284E-07 0.01621 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.72482E-07 0.00127  7.72119E-07 0.00127  8.76434E-07 0.01634 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43364E-03 0.00888  2.59162E-04 0.03258  8.07420E-04 0.01869  6.18472E-04 0.02064  1.36684E-03 0.01363  3.19351E-04 0.02674  6.23967E-05 0.06806 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.89152E-01 0.02358  1.24750E-02 6.4E-05  3.22503E-02 0.00023  1.06702E-01 0.00099  3.00739E-01 0.00054  1.25651E+00 0.00133  6.95651E+00 0.03114 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.47881E-07 0.01966  7.47509E-07 0.01966  8.47571E-07 0.05768 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.78893E-07 0.01965  6.78554E-07 0.01965  7.69481E-07 0.05769 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.96657E-03 0.04303  2.28497E-04 0.14449  6.92929E-04 0.08767  6.71868E-04 0.09279  1.04173E-03 0.06401  2.68950E-04 0.12652  6.26034E-05 0.31794 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.98165E-01 0.09828  1.24719E-02 0.00021  3.22380E-02 0.00077  1.06991E-01 0.00349  3.00067E-01 0.00194  1.25931E+00 0.00382  7.41409E+00 0.08667 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.02168E-03 0.04220  2.25814E-04 0.13799  7.00326E-04 0.08472  6.87673E-04 0.09065  1.07260E-03 0.06239  2.72588E-04 0.12193  6.26861E-05 0.32019 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.96172E-01 0.09556  1.24720E-02 0.00021  3.22400E-02 0.00075  1.07001E-01 0.00349  3.00113E-01 0.00193  1.25926E+00 0.00382  7.39041E+00 0.08715 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.99164E+03 0.03870 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.43762E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.65768E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47533E-03 0.00542 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.11925E+03 0.00544 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76157E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67449E-05 0.00941  6.67586E-05 0.00941  5.28136E-07 0.59996 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.74769E-05 0.03040  5.74140E-05 0.03038  9.78159E-07 0.64939 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.49541E-04 0.02507  1.49826E-04 0.02507  7.10668E-05 0.57606 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.66489E+01 0.01098 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46247E+01 0.00026  3.93381E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 14:19:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01307E+00  1.00702E+00  1.01223E+00  9.99030E-01  1.00232E+00  1.01090E+00  1.00117E+00  1.00864E+00  1.00020E+00  1.00407E+00  1.00092E+00  1.00985E+00  1.00086E+00  1.00347E+00  9.98613E-01  9.97535E-01  1.00406E+00  9.26047E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.58145E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.74185E-01 8.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.38612E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.52474E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.58752E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45476E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44544E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16672E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.16730E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333977 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33400E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33400E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.84514E+03 ;
RUNNING_TIME              (idx, 1)        =  5.58125E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.01638E+00  9.17833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.55672E+02  5.19047E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.27550E-01  1.39333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.94145E+01  7.65712E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.50510E+02  7.04095E+02 ];
CPU_USAGE                 (idx, 1)        = 8.68110 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99074E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.54820E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.93098E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74524E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.20742E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97072E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36183E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92613E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57511E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.57212E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75120E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.08064E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03868E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44222E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70280E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34259E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37156E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33796E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55634E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99912E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27077E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26987E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.05038E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41112E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09083E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.82069E+15 0.00031  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43599E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.33448E-02 -1.17783E+28  2.32575E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02549E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.54034E+18 0.00416  1.64423E-02 0.00411 ];
U233_FISS                 (idx, [1:   4]) = [  8.08030E+19 0.00058  8.62548E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.85667E+18 0.00191  7.31929E-02 0.00183 ];
U238_FISS                 (idx, [1:   4]) = [  1.04365E+14 0.49750  1.11646E-06 0.49752 ];
PU239_FISS                (idx, [1:   4]) = [  2.90873E+17 0.00907  3.10492E-03 0.00905 ];
PU240_FISS                (idx, [1:   4]) = [  1.99094E+16 0.03536  2.12610E-04 0.03542 ];
PU241_FISS                (idx, [1:   4]) = [  3.19176E+16 0.03051  3.40685E-04 0.03045 ];
TH232_CAPT                (idx, [1:   4]) = [  9.49247E+19 0.00053  6.63832E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17716E+19 0.00153  8.23224E-02 0.00149 ];
U235_CAPT                 (idx, [1:   4]) = [  2.39581E+18 0.00333  1.67549E-02 0.00333 ];
U238_CAPT                 (idx, [1:   4]) = [  1.04257E+15 0.15175  7.30298E-06 0.15174 ];
PU239_CAPT                (idx, [1:   4]) = [  1.34196E+17 0.01461  9.38473E-04 0.01461 ];
PU240_CAPT                (idx, [1:   4]) = [  5.32604E+16 0.02228  3.72440E-04 0.02226 ];
PU241_CAPT                (idx, [1:   4]) = [  6.22822E+15 0.06783  4.35706E-05 0.06786 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75793E+17 0.01170  1.22944E-03 0.01171 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001994 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09373E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001994 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5480148 5.48529E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3590466 3.59353E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931380 9.32115E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001994 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.72529E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36303E+20 4.3E-06  2.36303E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36380E+19 9.3E-08  9.36380E+19 9.3E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43005E+20 0.00027  1.28551E+20 0.00019  1.44534E+19 0.00185 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36643E+20 0.00016  2.22189E+20 0.00011  1.44534E+19 0.00185 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60690E+20 0.00031  2.60690E+20 0.00031  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44840E+22 0.00023  3.27503E+22 0.00020  1.73370E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.42999E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60943E+20 0.00018 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16009E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.24059E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.24059E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.78353E-01 0.05404 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.35133E-01 0.01255 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.74937E-04 0.02592 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.59119E+03 0.02940 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06792E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.69841E-01 0.04059 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.07478E-01 0.04059 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52358E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99967E+02 9.3E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.06676E-01 0.00045  1.00413E-01 0.00044  3.48699E-04 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06570E-01 0.00018 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06481E-01 0.00031 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06570E-01 0.00018 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99772E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84046E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84069E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.60484E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.60423E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.02483E-01 0.00141 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.02888E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.92290E-03 0.00566  2.82578E-04 0.01972  9.05925E-04 0.01198  7.08888E-04 0.01199  1.58834E-03 0.00899  3.57374E-04 0.01673  7.97992E-05 0.03902 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.09836E-01 0.01437  1.24778E-02 6.5E-05  3.22244E-02 0.00014  1.06712E-01 0.00056  3.01025E-01 0.00031  1.26022E+00 0.00087  6.54142E+00 0.02572 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44919E-03 0.00791  2.55586E-04 0.02744  7.95321E-04 0.01648  6.23948E-04 0.01803  1.39653E-03 0.01226  3.10422E-04 0.02632  6.73840E-05 0.05785 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00607E-01 0.02087  1.24781E-02 9.6E-05  3.22310E-02 0.00019  1.06638E-01 0.00074  3.01061E-01 0.00045  1.26009E+00 0.00108  7.19926E+00 0.01981 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.36900E-07 0.00134  8.36448E-07 0.00135  9.73787E-07 0.01708 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.58740E-07 0.00122  7.58330E-07 0.00123  8.82806E-07 0.01706 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45544E-03 0.00889  2.54522E-04 0.03406  7.85130E-04 0.01934  6.37990E-04 0.02189  1.39271E-03 0.01450  3.18152E-04 0.02871  6.69384E-05 0.06532 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.99720E-01 0.02397  1.24779E-02 6.1E-05  3.22300E-02 0.00024  1.06767E-01 0.00100  3.00941E-01 0.00053  1.26239E+00 0.00144  7.02567E+00 0.03085 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.39997E-07 0.01979  7.39544E-07 0.01980  8.55627E-07 0.04712 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.71137E-07 0.01980  6.70727E-07 0.01980  7.75572E-07 0.04704 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.05576E-03 0.04240  1.88509E-04 0.14244  6.78983E-04 0.08110  6.13278E-04 0.08126  1.23283E-03 0.06033  2.89124E-04 0.13777  5.30382E-05 0.28402 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.35812E-01 0.07656  1.24879E-02 0.00093  3.22300E-02 0.00089  1.06435E-01 0.00268  3.00707E-01 0.00199  1.27414E+00 0.00462  6.13815E+00 0.11056 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.07817E-03 0.04240  1.93844E-04 0.14038  6.80125E-04 0.08079  6.08685E-04 0.08087  1.25528E-03 0.05977  2.91428E-04 0.13613  4.88025E-05 0.28305 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.30525E-01 0.07164  1.24872E-02 0.00087  3.22269E-02 0.00087  1.06434E-01 0.00270  3.00627E-01 0.00197  1.27398E+00 0.00462  6.13815E+00 0.11056 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.15886E+03 0.03816 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.29136E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.51707E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44759E-03 0.00513 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.15901E+03 0.00523 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78566E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69052E-05 0.00820  6.69032E-05 0.00822  6.11887E-07 0.59369 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.76000E-05 0.02800  5.75160E-05 0.02799  1.08436E-06 0.69057 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.71141E-04 0.02501  1.71513E-04 0.02495  7.99489E-05 0.57575 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59314E+01 0.01080 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44544E+01 0.00026  3.89014E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 15:11:52 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02023E+00  1.01155E+00  1.00264E+00  1.00182E+00  1.00182E+00  1.00253E+00  1.00104E+00  1.01064E+00  1.00223E+00  9.98689E-01  9.98948E-01  1.00035E+00  1.01008E+00  1.00130E+00  1.00323E+00  1.00472E+00  1.00223E+00  9.25942E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.61774E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73823E-01 9.0E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37142E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51200E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59886E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45243E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44302E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20798E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.34393E-01 0.00058  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333997 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33420E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33420E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.29682E+03 ;
RUNNING_TIME              (idx, 1)        =  6.10397E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.10838E+00  9.20000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.07794E+02  5.21226E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.41283E-01  1.37333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.71573E+01  7.74277E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.02697E+02  7.06718E+02 ];
CPU_USAGE                 (idx, 1)        = 8.67766 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99073E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.54429E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.91699E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74001E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.06417E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92666E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33926E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95622E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57218E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53491E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.39262E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36013E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14346E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69253E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23825E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58127E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34571E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31706E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52844E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.03170E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59761E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23683E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03137E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.15384E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.02482E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.85157E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06820E-01 -2.35855E+28  2.44382E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02258E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.51414E+18 0.00416  1.61787E-02 0.00411 ];
U233_FISS                 (idx, [1:   4]) = [  7.93864E+19 0.00058  8.48265E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.12891E+18 0.00205  7.61732E-02 0.00195 ];
U238_FISS                 (idx, [1:   4]) = [  1.32231E+14 0.44423  1.40716E-06 0.44425 ];
PU239_FISS                (idx, [1:   4]) = [  5.99181E+17 0.00653  6.40235E-03 0.00649 ];
PU240_FISS                (idx, [1:   4]) = [  7.11306E+16 0.01863  7.60045E-04 0.01860 ];
PU241_FISS                (idx, [1:   4]) = [  1.19371E+17 0.01526  1.27531E-03 0.01520 ];
TH232_CAPT                (idx, [1:   4]) = [  9.31679E+19 0.00047  6.48049E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15554E+19 0.00153  8.03760E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49362E+18 0.00322  1.73449E-02 0.00320 ];
U238_CAPT                 (idx, [1:   4]) = [  2.61617E+15 0.09945  1.81753E-05 0.09940 ];
PU239_CAPT                (idx, [1:   4]) = [  2.78339E+17 0.00979  1.93604E-03 0.00978 ];
PU240_CAPT                (idx, [1:   4]) = [  1.82866E+17 0.01361  1.27171E-03 0.01355 ];
PU241_CAPT                (idx, [1:   4]) = [  2.52963E+16 0.03282  1.75912E-04 0.03280 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81240E+17 0.01130  1.26068E-03 0.01130 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002592 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09189E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002592 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5488518 5.49324E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3572983 3.57587E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 941091 9.41806E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002592 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 6.33299E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36611E+20 4.9E-06  2.36611E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35971E+19 1.0E-07  9.35971E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43743E+20 0.00029  1.28814E+20 0.00020  1.49290E+19 0.00194 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37340E+20 0.00018  2.22412E+20 0.00012  1.49290E+19 0.00194 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61719E+20 0.00033  2.61719E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.48333E+22 0.00023  3.30684E+22 0.00020  1.76491E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46498E+19 0.00158 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61990E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16404E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  8.00984E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.00984E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.03845E-01 0.05360 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17890E-01 0.01318 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.24104E-04 0.02332 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.94877E+03 0.03113 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.05823E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.92063E-01 0.03847 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.26853E-01 0.03847 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52797E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00054E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03858E-01 0.00046  1.00086E-01 0.00046  3.52955E-04 0.00878 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.04120E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.04095E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.04120E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98130E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81206E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.80979E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65114E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.65457E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09674E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10261E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96199E-03 0.00536  2.76498E-04 0.02111  9.25803E-04 0.01126  7.11959E-04 0.01302  1.60609E-03 0.00823  3.65388E-04 0.01797  7.62468E-05 0.03794 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.97997E-01 0.01439  1.24389E-02 0.00335  3.22041E-02 0.00015  1.06735E-01 0.00052  3.01246E-01 0.00030  1.25924E+00 0.00099  6.17511E+00 0.02810 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49406E-03 0.00717  2.48583E-04 0.02820  8.35142E-04 0.01448  6.15525E-04 0.01767  1.40914E-03 0.01164  3.14006E-04 0.02524  7.16585E-05 0.05701 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00849E-01 0.02111  1.24804E-02 0.00012  3.22077E-02 0.00022  1.06778E-01 0.00074  3.01220E-01 0.00040  1.25818E+00 0.00135  6.84364E+00 0.02262 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.34420E-07 0.00139  8.33946E-07 0.00139  9.61244E-07 0.02366 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.54142E-07 0.00129  7.53713E-07 0.00129  8.68937E-07 0.02375 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.51597E-03 0.00882  2.55047E-04 0.03336  8.21809E-04 0.02041  6.25876E-04 0.02079  1.42145E-03 0.01361  3.19264E-04 0.02765  7.25245E-05 0.06150 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07259E-01 0.02482  1.24810E-02 0.00018  3.22103E-02 0.00024  1.06558E-01 0.00086  3.01518E-01 0.00053  1.26213E+00 0.00176  6.84501E+00 0.03110 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.35573E-07 0.01991  7.34990E-07 0.01992  8.66014E-07 0.05155 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.65021E-07 0.01989  6.64491E-07 0.01989  7.83323E-07 0.05157 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.31353E-03 0.04028  2.47443E-04 0.12748  7.22765E-04 0.07988  5.21873E-04 0.09012  1.37850E-03 0.06196  3.32972E-04 0.12642  1.09970E-04 0.22528 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.74599E-01 0.12611  1.24775E-02 0.00015  3.22437E-02 0.00084  1.06724E-01 0.00325  3.01437E-01 0.00190  1.25702E+00 0.00534  7.28063E+00 0.07272 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.34934E-03 0.04001  2.45966E-04 0.12974  7.20188E-04 0.07929  5.37505E-04 0.08870  1.39508E-03 0.06186  3.38756E-04 0.12614  1.11843E-04 0.22030 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  5.78292E-01 0.12605  1.24780E-02 0.00016  3.22432E-02 0.00084  1.06720E-01 0.00326  3.01441E-01 0.00189  1.25714E+00 0.00534  7.28063E+00 0.07272 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.55547E+03 0.03577 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.26652E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.47131E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.57067E-03 0.00551 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.32016E+03 0.00556 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91390E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75628E-05 0.00687  6.75637E-05 0.00687  8.48953E-07 0.53082 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.79226E-05 0.02480  5.79250E-05 0.02479  5.20390E-07 0.55217 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.21667E-04 0.02275  2.22143E-04 0.02270  9.96773E-05 0.49815 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60043E+01 0.01246 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44302E+01 0.00029  3.86385E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 16:04:23 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02157E+00  1.00712E+00  1.00310E+00  9.93844E-01  1.00555E+00  1.00867E+00  1.00235E+00  1.00135E+00  1.00140E+00  1.00664E+00  9.98028E-01  1.00657E+00  1.00620E+00  1.00244E+00  1.00280E+00  1.00549E+00  1.00461E+00  9.22265E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.65331E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73467E-01 9.0E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35424E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49681E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61450E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45634E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44683E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.27001E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.53451E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334546 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33439E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33439E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.74977E+03 ;
RUNNING_TIME              (idx, 1)        =  6.62906E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.20083E+00  9.24500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.60153E+02  5.23582E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.54683E-01  1.34000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.50209E+01  7.86348E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.55086E+02  7.07314E+02 ];
CPU_USAGE                 (idx, 1)        = 8.67357 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99076E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.53967E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.91413E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73851E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.35004E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91495E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33344E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96485E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57101E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64238E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56950E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45997E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16276E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76480E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39507E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93222E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33879E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31200E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52133E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.05226E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00843E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22689E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02091E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.36414E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00624E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.86130E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93016E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61415E-01 -3.56399E+28  2.56436E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01802E+00 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.50865E+18 0.00405  1.61201E-02 0.00400 ];
U233_FISS                 (idx, [1:   4]) = [  7.92608E+19 0.00057  8.46926E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.08210E+18 0.00189  7.56775E-02 0.00189 ];
U238_FISS                 (idx, [1:   4]) = [  2.62860E+13 1.00000  2.81460E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  6.47214E+17 0.00691  6.91567E-03 0.00689 ];
PU240_FISS                (idx, [1:   4]) = [  8.33331E+16 0.01968  8.90394E-04 0.01966 ];
PU241_FISS                (idx, [1:   4]) = [  1.46467E+17 0.01353  1.56488E-03 0.01347 ];
TH232_CAPT                (idx, [1:   4]) = [  9.25117E+19 0.00059  6.43346E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15296E+19 0.00154  8.01786E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48958E+18 0.00313  1.73138E-02 0.00314 ];
U238_CAPT                 (idx, [1:   4]) = [  2.99125E+15 0.09648  2.08126E-05 0.09638 ];
PU239_CAPT                (idx, [1:   4]) = [  2.99140E+17 0.00932  2.08004E-03 0.00926 ];
PU240_CAPT                (idx, [1:   4]) = [  2.16109E+17 0.01110  1.50299E-03 0.01112 ];
PU241_CAPT                (idx, [1:   4]) = [  3.08848E+16 0.02865  2.14786E-04 0.02865 ];
SM149_CAPT                (idx, [1:   4]) = [  1.79474E+17 0.01293  1.24807E-03 0.01291 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003179 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07698E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003179 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5483116 5.48755E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3568886 3.57143E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 951177 9.51793E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003179 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.22820E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36679E+20 4.8E-06  2.36679E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35906E+19 1.0E-07  9.35906E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43741E+20 0.00026  1.28410E+20 0.00018  1.53306E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37331E+20 0.00016  2.22000E+20 0.00010  1.53306E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62043E+20 0.00032  2.62043E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.51417E+22 0.00022  3.33531E+22 0.00021  1.78863E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.49417E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62273E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16645E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  7.81090E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.81090E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.72783E-01 0.04243 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.08867E-01 0.01026 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.69240E-04 0.02019 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.24608E+03 0.02908 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.04827E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 9.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.98232E-01 0.02892 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.22272E-01 0.02892 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52887E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00069E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.03086E-01 0.00045  1.00007E-01 0.00045  3.44749E-04 0.00887 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.03397E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.03231E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.03397E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98447E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81085E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.80980E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65312E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.65456E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11175E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11110E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94925E-03 0.00559  2.82829E-04 0.02020  9.18719E-04 0.01125  7.14381E-04 0.01205  1.57769E-03 0.00782  3.67614E-04 0.01828  8.80176E-05 0.03342 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.18640E-01 0.01366  1.24814E-02 0.00011  3.22025E-02 0.00015  1.06835E-01 0.00050  3.01419E-01 0.00031  1.25844E+00 0.00102  6.37582E+00 0.02455 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46102E-03 0.00707  2.41598E-04 0.02792  8.02232E-04 0.01578  6.13709E-04 0.01727  1.40073E-03 0.01150  3.24930E-04 0.02461  7.78216E-05 0.04838 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.25560E-01 0.02040  1.24799E-02 9.2E-05  3.21995E-02 0.00021  1.06808E-01 0.00071  3.01251E-01 0.00043  1.25739E+00 0.00137  6.90098E+00 0.02069 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.42349E-07 0.00138  8.41932E-07 0.00139  9.60813E-07 0.01502 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.60675E-07 0.00133  7.60298E-07 0.00134  8.67735E-07 0.01503 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43186E-03 0.00903  2.45398E-04 0.03538  8.00926E-04 0.01835  6.21298E-04 0.02277  1.37687E-03 0.01390  3.09895E-04 0.03125  7.74753E-05 0.05603 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.18556E-01 0.02439  1.24789E-02 0.00016  3.22028E-02 0.00030  1.06943E-01 0.00108  3.01299E-01 0.00053  1.25656E+00 0.00204  6.94574E+00 0.02880 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.37273E-07 0.01968  7.36610E-07 0.01969  9.82423E-07 0.11269 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.65920E-07 0.01968  6.65320E-07 0.01968  8.87361E-07 0.11281 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.00715E-03 0.04493  2.15927E-04 0.14925  6.33637E-04 0.08087  6.03213E-04 0.09286  1.24247E-03 0.06970  2.67209E-04 0.12937  4.46882E-05 0.31542 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.55050E-01 0.08701  1.24760E-02 0.00018  3.21876E-02 0.00101  1.07236E-01 0.00347  3.00774E-01 0.00197  1.26301E+00 0.00536  7.63779E+00 0.08285 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97802E-03 0.04406  2.09741E-04 0.14675  6.33014E-04 0.08017  6.01704E-04 0.09110  1.22392E-03 0.06842  2.71199E-04 0.12222  3.84394E-05 0.30171 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.54280E-01 0.08341  1.24760E-02 0.00018  3.21941E-02 0.00099  1.07217E-01 0.00343  3.00782E-01 0.00196  1.26282E+00 0.00535  7.63779E+00 0.08285 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.11658E+03 0.04135 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.34889E-07 0.00082 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.53924E-07 0.00065 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40810E-03 0.00535 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.08211E+03 0.00529 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.07206E-09 0.00091 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.81960E-05 0.00578  6.81788E-05 0.00579  1.83553E-06 0.35545 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.17673E-05 0.02125  6.17987E-05 0.02127  1.41762E-06 0.45526 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.67204E-04 0.01945  2.67464E-04 0.01945  2.05517E-04 0.35035 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58680E+01 0.01033 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44683E+01 0.00027  3.86153E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 05:01:28 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 16:32:25 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650441688977 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02749E+00  1.01131E+00  1.00677E+00  1.00122E+00  1.00811E+00  9.94497E-01  1.01528E+00  1.00843E+00  1.01202E+00  1.01038E+00  1.00839E+00  1.00734E+00  1.00460E+00  1.00904E+00  1.00287E+00  9.98746E-01  1.00273E+00  8.70783E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.69596E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73040E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34167E-01 0.00020  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.48651E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62583E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46765E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45811E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.32868E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.74261E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333646 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33381E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33381E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.99588E+03 ;
RUNNING_TIME              (idx, 1)        =  6.90936E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.58817E-01  9.58817E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.29530E+00  9.44667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.88025E+02  2.78719E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.67717E-01  1.30333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.86701E+01  3.64922E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.87329E+02  6.87329E+02 ];
CPU_USAGE                 (idx, 1)        = 8.67790 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98488E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  8.54479E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 22044.15;
MEMSIZE                   (idx, 1)        = 21830.50;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 278.06;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 213.65;

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

TOT_ACTIVITY              (idx, 1)        =  6.90900E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73765E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.57976E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90712E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32869E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96757E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57065E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65637E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61597E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47174E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16301E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78577E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44107E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03203E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33680E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31055E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51929E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.05437E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13460E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22032E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01737E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.39866E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99087E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.87184E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57344E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16600E-01 -4.78246E+28  2.68621E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01389E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.51063E+18 0.00423  1.61442E-02 0.00415 ];
U233_FISS                 (idx, [1:   4]) = [  7.92442E+19 0.00057  8.46934E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  7.04560E+18 0.00187  7.53005E-02 0.00179 ];
U238_FISS                 (idx, [1:   4]) = [  1.57073E+14 0.40483  1.69487E-06 0.40483 ];
PU239_FISS                (idx, [1:   4]) = [  6.55421E+17 0.00623  7.00507E-03 0.00622 ];
PU240_FISS                (idx, [1:   4]) = [  8.66875E+16 0.01687  9.26557E-04 0.01688 ];
PU241_FISS                (idx, [1:   4]) = [  1.52420E+17 0.01376  1.62891E-03 0.01373 ];
TH232_CAPT                (idx, [1:   4]) = [  9.20767E+19 0.00051  6.39256E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15531E+19 0.00148  8.02103E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48248E+18 0.00320  1.72350E-02 0.00317 ];
U238_CAPT                 (idx, [1:   4]) = [  2.96326E+15 0.09556  2.05905E-05 0.09557 ];
PU239_CAPT                (idx, [1:   4]) = [  3.08452E+17 0.00932  2.14162E-03 0.00934 ];
PU240_CAPT                (idx, [1:   4]) = [  2.27555E+17 0.01059  1.57979E-03 0.01057 ];
PU241_CAPT                (idx, [1:   4]) = [  3.21045E+16 0.02973  2.22918E-04 0.02976 ];
SM149_CAPT                (idx, [1:   4]) = [  1.88260E+17 0.01148  1.30685E-03 0.01143 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001420 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07845E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001420 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5484073 5.48939E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3562641 3.56587E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 954706 9.55523E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001420 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -6.35162E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52483E-02 0.0E+00  3.52483E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36688E+20 5.0E-06  2.36688E+20 5.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35898E+19 9.5E-08  9.35898E+19 9.5E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44030E+20 0.00029  1.28108E+20 0.00018  1.59214E+19 0.00193 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37620E+20 0.00017  2.21698E+20 0.00011  1.59214E+19 0.00193 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.62395E+20 0.00035  2.62395E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54823E+22 0.00026  3.36570E+22 0.00023  1.82531E+21 0.00198 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50731E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.62693E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17098E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51105E+04 ;
TOT_FMASS                 (idx, 1)        =  7.63929E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51105E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.63929E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.45797E-01 0.03743 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.02698E-01 0.00944 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.49581E-04 0.01808 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.84936E+03 0.03311 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.04453E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.00865E-01 0.01892 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.14745E-01 0.01892 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52900E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 9.5E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.01804E-01 0.00045  9.98590E-02 0.00044  3.41231E-04 0.00943 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.01994E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.02064E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.01994E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.97280E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81301E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81383E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64953E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64793E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11150E-01 0.00132 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11024E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.95796E-03 0.00545  2.88196E-04 0.01964  9.20192E-04 0.01114  7.06857E-04 0.01213  1.58837E-03 0.00891  3.74026E-04 0.01761  8.03181E-05 0.04051 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.07710E-01 0.01493  1.24805E-02 0.00012  3.21963E-02 0.00016  1.06795E-01 0.00053  3.01274E-01 0.00032  1.25772E+00 0.00109  6.21392E+00 0.02756 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47757E-03 0.00779  2.53769E-04 0.02959  8.14922E-04 0.01648  6.23372E-04 0.01858  1.38811E-03 0.01290  3.29514E-04 0.02547  6.78826E-05 0.05385 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02833E-01 0.02012  1.24809E-02 0.00014  3.21928E-02 0.00025  1.06831E-01 0.00074  3.01107E-01 0.00042  1.25835E+00 0.00132  6.99102E+00 0.02029 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.56370E-07 0.00181  8.55965E-07 0.00182  9.80102E-07 0.02244 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.72220E-07 0.00173  7.71855E-07 0.00173  8.83641E-07 0.02231 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39923E-03 0.00945  2.38723E-04 0.03641  7.95594E-04 0.01939  6.00648E-04 0.02178  1.36885E-03 0.01512  3.22040E-04 0.02937  7.33746E-05 0.06502 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.13909E-01 0.02425  1.24824E-02 0.00030  3.22029E-02 0.00030  1.06719E-01 0.00097  3.01337E-01 0.00054  1.25922E+00 0.00191  6.98061E+00 0.02862 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.46237E-07 0.01971  7.45905E-07 0.01971  7.93319E-07 0.05532 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.73307E-07 0.01970  6.73008E-07 0.01970  7.15886E-07 0.05532 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85784E-03 0.04565  2.29241E-04 0.13844  6.28574E-04 0.08599  5.70685E-04 0.09444  1.08722E-03 0.06827  3.12227E-04 0.12022  2.98879E-05 0.30785 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.95679E-01 0.11837  1.24757E-02 0.00016  3.21331E-02 0.00123  1.06698E-01 0.00314  3.01818E-01 0.00213  1.25520E+00 0.00625  6.99422E+00 0.11280 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.84342E-03 0.04531  2.30811E-04 0.13739  6.31679E-04 0.08401  5.58978E-04 0.09527  1.09273E-03 0.06673  3.00470E-04 0.11921  2.87473E-05 0.30129 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.85104E-01 0.11699  1.24757E-02 0.00016  3.21360E-02 0.00123  1.06715E-01 0.00314  3.01799E-01 0.00212  1.25470E+00 0.00637  6.99422E+00 0.11280 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.83698E+03 0.04131 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.46436E-07 0.00087 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.63268E-07 0.00071 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38154E-03 0.00646 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.99434E+03 0.00630 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.26460E-09 0.00098 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.95681E-05 0.00514  6.95579E-05 0.00514  2.38737E-06 0.31749 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.08477E-05 0.01677  6.08501E-05 0.01678  1.77657E-06 0.34662 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.45872E-04 0.01745  3.46241E-04 0.01747  2.51933E-04 0.31240 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63419E+01 0.01077 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45811E+01 0.00027  3.86324E+01 0.00038 ];

