
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 08:16:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03860E+00  9.85991E-01  9.90903E-01  9.87605E-01  1.01039E+00  9.94952E-01  1.00387E+00  1.00432E+00  1.00801E+00  9.95958E-01  9.86795E-01  1.01171E+00  9.97062E-01  9.98616E-01  9.79656E-01  1.01497E+00  9.93555E-01  9.97023E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.68671E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73133E-01 9.0E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31189E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45645E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70301E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45323E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44321E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.41173E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.88366E-01 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500345 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50031E+04 0.00053 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50031E+04 0.00053 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.88935E+02 ;
RUNNING_TIME              (idx, 1)        =  2.00553E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.18783E-01  1.18783E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.87893E+01  1.87893E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.00783E-01  5.24833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.00516E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.42069 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97696E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.39131E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  2.73190E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.14841E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.05432E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.73190E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.14841E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.62569E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.39381E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.62569E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.39381E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.23841E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.05337E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07893E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.57870E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15174E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.64703E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.53079E+18 0.00408  1.63287E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  9.22116E+19 0.00049  9.83671E-01 6.7E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.12841E+19 0.00056  7.48349E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33550E+19 0.00135  1.09490E-01 0.00135 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001231 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15602E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001231 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5088443 5.09384E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3910701 3.91466E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1002087 1.00306E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001231 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.86265E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36480E+20 3.8E-06  2.36480E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37767E+19 1.2E-07  9.37767E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.21859E+20 0.00032  1.07633E+20 0.00023  1.42259E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.15636E+20 0.00018  2.01410E+20 0.00012  1.42259E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.39467E+20 0.00037  2.39467E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.28264E+22 0.00026  3.10918E+22 0.00024  1.73460E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40210E+19 0.00136 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39657E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06517E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29426E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.21155E-01 0.07293 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.84584E-01 0.00811 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.64193E-04 0.01436 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.67182E+03 0.02074 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99704E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.22334E-01 0.06328 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.79984E-01 0.06328 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52173E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.87250E-01 0.00040  6.14983E-02 0.00038  2.00011E-04 0.00878 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.87894E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.87577E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.87894E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09808E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00661E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00489E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35930E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36142E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51100E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51590E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.29413E-03 0.00523  2.69593E-04 0.01912  8.17783E-04 0.01101  5.85761E-04 0.01280  1.29905E-03 0.00855  2.75105E-04 0.01857  4.68388E-05 0.04435 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.37176E-01 0.01359  1.24755E-02 3.0E-05  3.23406E-02 9.0E-05  1.06139E-01 0.00063  2.97174E-01 0.00026  1.23577E+00 0.00023  4.39179E+00 0.04288 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22810E-03 0.00822  2.57027E-04 0.02756  8.08444E-04 0.01655  5.75361E-04 0.01916  1.27150E-03 0.01281  2.69660E-04 0.03020  4.61158E-05 0.06336 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.36286E-01 0.02048  1.24760E-02 3.7E-05  3.23427E-02 0.00013  1.06108E-01 0.00089  2.97101E-01 0.00037  1.23593E+00 0.00029  6.29429E+00 0.02944 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.66261E-07 0.00126  8.65921E-07 0.00126  9.70274E-07 0.01636 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.55164E-07 0.00120  8.54829E-07 0.00120  9.57790E-07 0.01637 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24381E-03 0.00896  2.61684E-04 0.03131  8.03621E-04 0.01877  5.72155E-04 0.02156  1.29885E-03 0.01379  2.64987E-04 0.03129  4.25128E-05 0.07336 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.26818E-01 0.02293  1.24745E-02 6.2E-05  3.23395E-02 0.00016  1.06100E-01 0.00109  2.97176E-01 0.00042  1.23624E+00 0.00038  5.95255E+00 0.04476 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.71001E-07 0.01716  7.70586E-07 0.01716  8.52004E-07 0.05456 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.61160E-07 0.01714  7.60748E-07 0.01715  8.41609E-07 0.05467 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87799E-03 0.04162  2.33063E-04 0.13070  7.19602E-04 0.07275  5.24593E-04 0.09625  1.12614E-03 0.05863  2.21394E-04 0.13256  5.31972E-05 0.24961 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.62267E-01 0.08978  1.24746E-02 0.00016  3.23282E-02 0.00043  1.06040E-01 0.00314  2.96705E-01 0.00143  1.23583E+00 0.00107  6.44147E+00 0.11695 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89660E-03 0.04108  2.32157E-04 0.12736  7.23394E-04 0.07245  5.32637E-04 0.09503  1.13295E-03 0.05790  2.27297E-04 0.13396  4.81638E-05 0.25133 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.51150E-01 0.08625  1.24747E-02 0.00015  3.23283E-02 0.00042  1.06022E-01 0.00312  2.96748E-01 0.00143  1.23571E+00 0.00108  6.44147E+00 0.11695 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75871E+03 0.03832 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.59332E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.48321E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28500E-03 0.00523 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82288E+03 0.00520 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.24041E-09 0.00129 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.27200E-05 0.00423  7.26995E-05 0.00422  3.06816E-06 0.27280 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.88832E-05 0.01408  7.88579E-05 0.01415  3.52189E-06 0.34604 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.37838E-04 0.01378  5.38108E-04 0.01381  4.72640E-04 0.25467 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74296E+01 0.01103 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44321E+01 0.00026  4.00933E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 08:46:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02371E+00  9.89023E-01  1.00522E+00  9.94025E-01  1.00712E+00  1.00060E+00  1.00403E+00  1.00537E+00  9.93039E-01  1.00950E+00  9.90473E-01  1.00490E+00  9.89852E-01  9.98067E-01  9.92445E-01  9.99705E-01  1.00292E+00  9.89990E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.68119E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73188E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31189E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45613E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70322E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46009E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45004E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.42583E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.88755E-01 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500721 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50062E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50062E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.90686E+02 ;
RUNNING_TIME              (idx, 1)        =  5.03732E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.05033E-01  8.62500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.89991E+01  3.02098E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.57500E-02  1.57500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.50600E-01  4.97667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.03695E+01  2.95956E+02 ];
CPU_USAGE                 (idx, 1)        = 9.74102 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98342E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.73726E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  5.75341E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60224E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65255E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.45931E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.02827E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26325E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46846E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.42678E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.64926E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.36444E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.71874E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06233E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17738E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44212E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35437E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17472E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55248E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.61318E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38128E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.38232E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51315E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.53641E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.25965E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.67988E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26594E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.11205E-04 -2.39284E+25  2.15198E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.75241E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.55187E+18 0.00395  1.65494E-02 0.00387 ];
U233_FISS                 (idx, [1:   4]) = [  9.21836E+19 0.00051  9.83142E-01 6.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.10911E+14 0.23764  4.39904E-06 0.23765 ];
TH232_CAPT                (idx, [1:   4]) = [  9.23265E+19 0.00055  7.43600E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33739E+19 0.00129  1.07717E-01 0.00125 ];
U235_CAPT                 (idx, [1:   4]) = [  1.21115E+14 0.44499  9.69917E-07 0.44501 ];
SM149_CAPT                (idx, [1:   4]) = [  1.32901E+16 0.04473  1.06998E-04 0.04472 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002481 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13397E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002481 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5126087 5.13073E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3871280 3.87464E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1005114 1.00597E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002481 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.06754E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36470E+20 3.8E-06  2.36470E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.2E-07  9.37764E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.24107E+20 0.00031  1.09695E+20 0.00023  1.44119E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17883E+20 0.00018  2.03471E+20 0.00013  1.44119E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41997E+20 0.00036  2.41997E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32009E+22 0.00026  3.14421E+22 0.00023  1.75875E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43452E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42228E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07806E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29516E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29516E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.78764E-01 0.06734 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.94717E-01 0.00848 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.68310E-04 0.01618 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.29805E+03 0.02173 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99413E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.86208E-01 0.05563 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.37225E-01 0.05563 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52163E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77000E-01 0.00043  6.08693E-02 0.00042  1.98308E-04 0.00838 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.77357E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.77210E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.77357E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08669E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00741E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00768E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35824E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.35761E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52424E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51806E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34002E-03 0.00558  2.68008E-04 0.01936  8.06963E-04 0.01140  6.12581E-04 0.01298  1.31822E-03 0.00849  2.78664E-04 0.01903  5.55780E-05 0.04471 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53280E-01 0.01510  1.24451E-02 0.00251  3.23373E-02 9.3E-05  1.06093E-01 0.00060  2.97282E-01 0.00026  1.23587E+00 0.00022  4.58669E+00 0.04076 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25180E-03 0.00810  2.54645E-04 0.02892  7.93711E-04 0.01685  5.94405E-04 0.01958  1.28150E-03 0.01344  2.69478E-04 0.02927  5.80533E-05 0.06616 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.58265E-01 0.02278  1.24769E-02 2.7E-05  3.23352E-02 0.00013  1.06119E-01 0.00085  2.97398E-01 0.00040  1.23611E+00 0.00027  6.32322E+00 0.02842 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.75367E-07 0.00127  8.75019E-07 0.00127  9.85697E-07 0.02769 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.55171E-07 0.00120  8.54831E-07 0.00119  9.63346E-07 0.02799 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24816E-03 0.00862  2.69682E-04 0.03091  7.82787E-04 0.01844  5.96179E-04 0.02092  1.28063E-03 0.01390  2.68161E-04 0.03062  5.07177E-05 0.07324 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.48590E-01 0.02368  1.24763E-02 4.0E-05  3.23425E-02 0.00015  1.06014E-01 0.00099  2.97338E-01 0.00044  1.23577E+00 0.00037  6.53211E+00 0.03911 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.80803E-07 0.01727  7.80452E-07 0.01727  8.70378E-07 0.05117 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.62862E-07 0.01726  7.62519E-07 0.01726  8.50106E-07 0.05110 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89343E-03 0.04372  2.67090E-04 0.13099  7.27857E-04 0.07380  5.39802E-04 0.09375  1.10475E-03 0.06594  2.00784E-04 0.14979  5.31411E-05 0.25669 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.02913E-01 0.12312  1.24747E-02 0.00015  3.23435E-02 0.00052  1.05780E-01 0.00285  2.97827E-01 0.00162  1.23484E+00 0.00134  8.11312E+00 0.08383 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89881E-03 0.04320  2.62450E-04 0.12886  7.34334E-04 0.07198  5.35478E-04 0.09313  1.10884E-03 0.06584  2.01270E-04 0.15011  5.64380E-05 0.25712 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.00411E-01 0.12183  1.24747E-02 0.00015  3.23444E-02 0.00052  1.05766E-01 0.00282  2.97850E-01 0.00162  1.23490E+00 0.00134  8.11312E+00 0.08383 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.72165E+03 0.04064 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.71237E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.51139E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.27661E-03 0.00534 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.76190E+03 0.00542 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.14281E-09 0.00128 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.16887E-05 0.00435  7.17004E-05 0.00435  1.93083E-06 0.29833 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.49011E-05 0.01510  7.49599E-05 0.01512  1.40168E-06 0.45410 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.46201E-04 0.01575  4.46489E-04 0.01570  3.64777E-04 0.28687 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70420E+01 0.01181 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45004E+01 0.00028  4.01268E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 09:18:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02121E+00  9.99537E-01  1.00293E+00  9.92509E-01  1.00075E+00  9.95781E-01  9.98760E-01  1.01290E+00  1.00281E+00  1.00271E+00  9.96088E-01  9.96598E-01  9.91235E-01  1.00729E+00  9.84781E-01  9.99348E-01  9.98675E-01  9.96082E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.66633E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73337E-01 9.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31045E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45400E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69998E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47396E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46396E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.46016E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.85371E-01 0.00059  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500699 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50049E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50049E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.11473E+02 ;
RUNNING_TIME              (idx, 1)        =  8.25988E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.93483E-01  8.84500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  8.11121E+01  3.21130E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.17333E-02  1.59833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.99850E-01  4.92000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.25950E+01  4.46366E+02 ];
CPU_USAGE                 (idx, 1)        = 9.82427 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98378E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.82706E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.36478E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65129E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.23709E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.84166E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.28774E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49200E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49131E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.71485E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.38891E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.82128E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  7.90752E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89335E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.59816E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.48968E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34019E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32999E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53746E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.75103E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17457E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.99151E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.50709E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.34720E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.46811E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.79911E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61904E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.94356E-04 -1.27890E+26  2.15302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.92394E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.58136E+18 0.00411  1.68634E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  9.20533E+19 0.00051  9.81743E-01 7.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.16805E+16 0.04741  1.24634E-04 0.04744 ];
TH232_CAPT                (idx, [1:   4]) = [  9.37735E+19 0.00055  7.38588E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33888E+19 0.00134  1.05457E-01 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  3.91154E+15 0.07761  3.08644E-05 0.07764 ];
SM149_CAPT                (idx, [1:   4]) = [  6.41993E+16 0.01978  5.05596E-04 0.01975 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001961 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13576E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001961 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5177630 5.18267E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3824054 3.82755E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1000277 1.00114E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001961 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.72529E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36455E+20 3.6E-06  2.36455E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37758E+19 1.1E-07  9.37758E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.27008E+20 0.00030  1.12462E+20 0.00022  1.45454E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.20784E+20 0.00017  2.06238E+20 0.00012  1.45454E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.44978E+20 0.00037  2.44978E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36830E+22 0.00025  3.19141E+22 0.00023  1.76886E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.45270E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.45311E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09474E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29736E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29736E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.15194E-01 0.04857 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.05594E-01 0.01104 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.83991E-04 0.01896 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.15862E+03 0.02382 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99891E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.02231E-01 0.03634 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.31879E-01 0.03634 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52150E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.65024E-01 0.00043  6.01219E-02 0.00042  1.98230E-04 0.00879 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.65019E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.65264E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.65019E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07235E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01899E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02025E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34264E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34065E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52814E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52330E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.36800E-03 0.00599  2.81135E-04 0.01992  8.30756E-04 0.01123  6.08296E-04 0.01297  1.32186E-03 0.00898  2.76104E-04 0.01923  4.98520E-05 0.04504 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.36914E-01 0.01440  1.24450E-02 0.00251  3.23413E-02 0.00010  1.06120E-01 0.00058  2.97475E-01 0.00028  1.23499E+00 0.00026  4.23060E+00 0.04316 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24186E-03 0.00834  2.63489E-04 0.02938  7.96875E-04 0.01655  5.80914E-04 0.01955  1.27968E-03 0.01236  2.75196E-04 0.02864  4.57138E-05 0.06620 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.34059E-01 0.01848  1.24761E-02 3.3E-05  3.23334E-02 0.00013  1.06129E-01 0.00085  2.97339E-01 0.00037  1.23496E+00 0.00031  5.89364E+00 0.03071 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.93130E-07 0.00122  8.92897E-07 0.00123  9.60767E-07 0.01513 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.61835E-07 0.00117  8.61612E-07 0.00117  9.26896E-07 0.01510 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27733E-03 0.00899  2.72385E-04 0.03272  7.98592E-04 0.01849  5.96330E-04 0.02079  1.29921E-03 0.01416  2.61734E-04 0.03221  4.90767E-05 0.07432 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.36245E-01 0.02554  1.24768E-02 3.4E-05  3.23397E-02 0.00017  1.06054E-01 0.00092  2.97342E-01 0.00044  1.23550E+00 0.00043  5.93666E+00 0.04347 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.90223E-07 0.01706  7.90109E-07 0.01705  8.37550E-07 0.05822 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.62854E-07 0.01705  7.62745E-07 0.01705  8.08482E-07 0.05806 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93890E-03 0.04414  2.89299E-04 0.12614  8.01197E-04 0.08444  5.04881E-04 0.08903  1.05021E-03 0.06359  2.51044E-04 0.13785  4.22662E-05 0.31952 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.07425E-01 0.07450  1.24743E-02 0.00015  3.23444E-02 0.00054  1.05853E-01 0.00289  2.97237E-01 0.00149  1.23294E+00 0.00122  5.80444E+00 0.13385 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94719E-03 0.04338  2.98875E-04 0.12588  7.93433E-04 0.08228  5.00165E-04 0.08577  1.05716E-03 0.06184  2.53189E-04 0.13425  4.43632E-05 0.31204 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.14695E-01 0.07882  1.24744E-02 0.00015  3.23457E-02 0.00054  1.05897E-01 0.00291  2.97276E-01 0.00148  1.23294E+00 0.00122  5.79873E+00 0.13372 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.72744E+03 0.04121 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.85471E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.54435E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29892E-03 0.00566 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.72600E+03 0.00567 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95596E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.86489E-05 0.00550  6.86542E-05 0.00552  8.25483E-07 0.44994 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.93238E-05 0.02502  6.93297E-05 0.02502  9.18575E-07 0.60890 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.73279E-04 0.01840  2.73703E-04 0.01844  1.55349E-04 0.44759 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73284E+01 0.01152 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46396E+01 0.00028  4.03402E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 09:51:30 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02715E+00  9.98450E-01  9.99175E-01  9.92114E-01  9.89475E-01  1.00173E+00  9.97817E-01  1.00158E+00  1.00149E+00  1.00397E+00  9.84856E-01  1.00373E+00  9.95073E-01  1.01020E+00  9.88142E-01  1.00538E+00  1.00116E+00  9.98503E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.65742E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73426E-01 8.1E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31228E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45534E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69593E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48814E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47813E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.48196E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.84398E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500499 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50043E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50043E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.13789E+03 ;
RUNNING_TIME              (idx, 1)        =  1.15395E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.86383E-01  9.29000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.13788E+02  3.26759E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.75667E-02  1.58333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.52367E-01  5.24667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.15391E+02  4.69830E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86084 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98394E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86646E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.48253E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66667E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.17834E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88175E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31483E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.56929E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50362E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.77299E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.55037E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.98239E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.11197E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.78949E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.73915E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48702E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.35482E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34117E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55247E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.40671E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42475E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.06483E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53415E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.21564E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.63247E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  9.89876E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.16546E-03 -2.50776E+26  2.15425E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.09645E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.59187E+18 0.00406  1.69871E-02 0.00406 ];
U233_FISS                 (idx, [1:   4]) = [  9.18507E+19 0.00049  9.80102E-01 7.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.46752E+16 0.02432  4.76741E-04 0.02431 ];
TH232_CAPT                (idx, [1:   4]) = [  9.50807E+19 0.00052  7.35704E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33498E+19 0.00141  1.03299E-01 0.00139 ];
U235_CAPT                 (idx, [1:   4]) = [  1.64023E+16 0.03830  1.26978E-04 0.03832 ];
SM149_CAPT                (idx, [1:   4]) = [  9.93733E+16 0.01623  7.68985E-04 0.01623 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001701 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16032E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001701 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5217126 5.22241E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3783335 3.78700E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1001240 1.00219E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001701 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.81259E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36449E+20 4.0E-06  2.36449E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 1.3E-07  9.37751E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29260E+20 0.00032  1.14458E+20 0.00023  1.48026E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23036E+20 0.00018  2.08233E+20 0.00013  1.48026E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.47469E+20 0.00035  2.47469E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.40790E+22 0.00025  3.22848E+22 0.00023  1.79418E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.48024E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.47838E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10941E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29692E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29692E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.57137E-01 0.04956 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.16960E-01 0.01256 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.23056E-04 0.02181 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.03756E+03 0.02457 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99785E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.84405E-01 0.03714 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.15730E-01 0.03714 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52145E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.54872E-01 0.00043  5.94858E-02 0.00042  1.93101E-04 0.00879 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.55169E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.55517E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.55169E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06152E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02462E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02445E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33499E-02 0.00127 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33501E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53546E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53642E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.40401E-03 0.00579  2.75798E-04 0.01962  8.26926E-04 0.01115  6.30749E-04 0.01347  1.32924E-03 0.00904  2.89504E-04 0.01934  5.17869E-05 0.04643 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.49150E-01 0.01468  1.24750E-02 3.4E-05  3.23347E-02 9.6E-05  1.06102E-01 0.00058  2.97546E-01 0.00027  1.23324E+00 0.00252  4.54298E+00 0.04240 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25563E-03 0.00853  2.66667E-04 0.02936  7.82795E-04 0.01689  5.99158E-04 0.01926  1.27709E-03 0.01383  2.77663E-04 0.02751  5.22649E-05 0.07029 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.56708E-01 0.02340  1.24747E-02 4.5E-05  3.23377E-02 0.00015  1.06166E-01 0.00083  2.97541E-01 0.00039  1.23616E+00 0.00031  6.61150E+00 0.02799 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.04600E-07 0.00120  9.04212E-07 0.00120  1.01928E-06 0.02038 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.63708E-07 0.00111  8.63339E-07 0.00111  9.73226E-07 0.02037 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23516E-03 0.00884  2.59376E-04 0.03218  7.92565E-04 0.01892  5.98325E-04 0.01982  1.26089E-03 0.01435  2.71213E-04 0.03079  5.27941E-05 0.07082 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53251E-01 0.02497  1.24753E-02 5.0E-05  3.23250E-02 0.00014  1.06141E-01 0.00109  2.97485E-01 0.00047  1.23602E+00 0.00043  6.46786E+00 0.04077 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.99867E-07 0.01712  7.99492E-07 0.01712  8.37281E-07 0.06057 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.64237E-07 0.01712  7.63879E-07 0.01712  7.99849E-07 0.06034 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.75795E-03 0.04399  2.50926E-04 0.12594  7.00399E-04 0.07747  5.29333E-04 0.09308  1.01700E-03 0.06031  2.19328E-04 0.13470  4.09688E-05 0.28970 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.64560E-01 0.12868  1.24782E-02 6.8E-05  3.23267E-02 0.00051  1.05656E-01 0.00269  2.97071E-01 0.00155  1.23554E+00 0.00130  6.49064E+00 0.12315 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.74923E-03 0.04309  2.54059E-04 0.12478  6.95212E-04 0.07543  5.22239E-04 0.09060  1.01670E-03 0.05958  2.17731E-04 0.13699  4.32900E-05 0.27495 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.66024E-01 0.12688  1.24782E-02 6.9E-05  3.23271E-02 0.00051  1.05683E-01 0.00273  2.96970E-01 0.00151  1.23556E+00 0.00130  6.49064E+00 0.12315 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.46627E+03 0.04106 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.99810E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.59138E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22867E-03 0.00571 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.58862E+03 0.00571 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89412E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.87472E-05 0.00732  6.87453E-05 0.00731  8.80887E-07 0.45713 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.11052E-05 0.02665  6.10283E-05 0.02667  1.17905E-06 0.58092 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.12775E-04 0.02138  2.12997E-04 0.02141  1.41242E-04 0.44563 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72180E+01 0.01206 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47813E+01 0.00027  4.04900E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 10:25:32 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01681E+00  9.93130E-01  1.00276E+00  9.92385E-01  1.00713E+00  1.00437E+00  1.00967E+00  1.00411E+00  9.98363E-01  9.99317E-01  9.99487E-01  9.97847E-01  9.94522E-01  1.00688E+00  9.81122E-01  1.00279E+00  9.97116E-01  9.92189E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.65078E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73492E-01 8.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31036E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45316E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69193E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50487E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49486E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.52187E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.84501E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500608 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50044E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50044E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.47680E+03 ;
RUNNING_TIME              (idx, 1)        =  1.49439E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.78533E-01  9.21500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.47711E+02  3.39227E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.29167E-02  1.53500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.01267E-01  4.88333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.49435E+02  4.77269E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88229 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98459E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88943E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.50827E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65978E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.58933E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90905E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33228E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.56740E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49467E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.96209E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.66923E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.26941E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.23413E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.68688E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.84570E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93291E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.31244E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30862E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50779E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.36319E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81136E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.06111E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.50195E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.53617E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.68326E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.00589E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64038E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.37781E-03 -5.11642E+26  2.15685E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.36216E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59834E+18 0.00394  1.70414E-02 0.00386 ];
U233_FISS                 (idx, [1:   4]) = [  9.15798E+19 0.00052  9.76492E-01 7.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.69205E+17 0.01239  1.80436E-03 0.01240 ];
TH232_CAPT                (idx, [1:   4]) = [  9.71329E+19 0.00051  7.31617E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33352E+19 0.00127  1.00442E-01 0.00119 ];
U235_CAPT                 (idx, [1:   4]) = [  5.89866E+16 0.02028  4.44335E-04 0.02030 ];
SM149_CAPT                (idx, [1:   4]) = [  1.45291E+17 0.01337  1.09420E-03 0.01333 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001741 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13993E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001741 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5274308 5.27957E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3725927 3.72947E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1001506 1.00236E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001741 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.60187E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36434E+20 3.8E-06  2.36434E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32678E+20 0.00029  1.17638E+20 0.00022  1.50403E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26451E+20 0.00017  2.11411E+20 0.00012  1.50403E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.51471E+20 0.00036  2.51471E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47217E+22 0.00024  3.28997E+22 0.00022  1.82192E+21 0.00169 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.52081E+19 0.00156 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.51659E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.13154E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29295E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29295E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.67461E-01 0.05264 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17066E-01 0.01336 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.95348E-04 0.02414 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.91026E+03 0.02570 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99767E-01 0.00016 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.13688E-01 0.04195 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.52267E-01 0.04195 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52134E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.40293E-01 0.00046  5.85787E-02 0.00045  1.91119E-04 0.00883 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.40597E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.40248E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.40597E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04543E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02744E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02673E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33136E-02 0.00145 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33197E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.55042E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.55931E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.46147E-03 0.00577  2.74375E-04 0.02002  8.46593E-04 0.01131  6.29334E-04 0.01312  1.37188E-03 0.00876  2.85020E-04 0.01968  5.42664E-05 0.04457 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.56583E-01 0.01647  1.24437E-02 0.00251  3.23258E-02 9.4E-05  1.06276E-01 0.00058  2.97523E-01 0.00027  1.23355E+00 0.00252  4.83971E+00 0.04070 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24450E-03 0.00862  2.60397E-04 0.03004  7.93725E-04 0.01624  5.95123E-04 0.02058  1.28211E-03 0.01355  2.62517E-04 0.02663  5.06252E-05 0.06277 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.60517E-01 0.02258  1.24753E-02 4.1E-05  3.23258E-02 0.00013  1.06191E-01 0.00080  2.97419E-01 0.00038  1.23651E+00 0.00034  6.86678E+00 0.02671 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.21313E-07 0.00132  9.21038E-07 0.00133  1.00885E-06 0.01694 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.66219E-07 0.00121  8.65959E-07 0.00122  9.48660E-07 0.01696 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24631E-03 0.00893  2.55753E-04 0.03266  7.96953E-04 0.01857  5.94103E-04 0.02162  1.28874E-03 0.01459  2.59814E-04 0.03344  5.09511E-05 0.07838 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.57494E-01 0.02693  1.24749E-02 5.8E-05  3.23255E-02 0.00016  1.06233E-01 0.00107  2.97475E-01 0.00044  1.23596E+00 0.00048  7.29021E+00 0.03567 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.14682E-07 0.01710  8.14307E-07 0.01711  8.54013E-07 0.05927 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.66188E-07 0.01710  7.65833E-07 0.01711  8.03773E-07 0.05937 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.97986E-03 0.04337  2.13445E-04 0.15017  7.46574E-04 0.07678  5.36783E-04 0.10170  1.17090E-03 0.06961  2.61061E-04 0.13777  5.10967E-05 0.28586 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.54614E-01 0.08702  1.24726E-02 0.00020  3.23258E-02 0.00058  1.06215E-01 0.00341  2.96373E-01 0.00131  1.23629E+00 0.00107  7.91215E+00 0.09237 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00410E-03 0.04254  2.12311E-04 0.14679  7.58547E-04 0.07513  5.34170E-04 0.10138  1.18204E-03 0.06782  2.62157E-04 0.13704  5.48690E-05 0.27641 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.54623E-01 0.08642  1.24726E-02 0.00020  3.23267E-02 0.00059  1.06185E-01 0.00335  2.96431E-01 0.00133  1.23610E+00 0.00107  7.91215E+00 0.09237 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.67554E+03 0.03973 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.14038E-07 0.00081 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.59382E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28864E-03 0.00694 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.59805E+03 0.00691 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.87304E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70924E-05 0.00895  6.70862E-05 0.00896  1.43321E-06 0.36027 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.73279E-05 0.02638  5.73271E-05 0.02643  1.32517E-06 0.45333 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.86335E-04 0.02338  1.86177E-04 0.02339  2.30223E-04 0.35203 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71599E+01 0.01164 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49486E+01 0.00027  4.06143E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 11:00:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02251E+00  1.00213E+00  1.00240E+00  9.96933E-01  1.00549E+00  9.97626E-01  9.89792E-01  1.00408E+00  1.01016E+00  1.00006E+00  9.93562E-01  9.98664E-01  9.96842E-01  1.00324E+00  9.94810E-01  9.98906E-01  9.94477E-01  9.88329E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.64730E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73527E-01 8.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30766E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45036E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69292E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51947E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50943E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.56007E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.86030E-01 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500583 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50019E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50019E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.82091E+03 ;
RUNNING_TIME              (idx, 1)        =  1.84008E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.72083E-01  9.35500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.82153E+02  3.44426E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.86500E-02  1.57333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.51083E-01  4.97667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.84004E+02  4.90292E+02 ];
CPU_USAGE                 (idx, 1)        = 9.89581 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98444E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90393E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.58645E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67081E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.75239E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95198E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36099E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.60210E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50230E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09594E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.78389E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.70148E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.42896E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.24288E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94070E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34589E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32443E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31738E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51977E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.03732E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.16777E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10816E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53867E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.40418E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.80317E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.01810E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96057E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.57042E-03 -7.68261E+26  2.15942E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.59213E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.62031E+18 0.00406  1.72834E-02 0.00399 ];
U233_FISS                 (idx, [1:   4]) = [  9.11271E+19 0.00054  9.72097E-01 9.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.60730E+17 0.00825  3.84836E-03 0.00826 ];
TH232_CAPT                (idx, [1:   4]) = [  9.86930E+19 0.00053  7.28457E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32948E+19 0.00137  9.81309E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  1.25772E+17 0.01438  9.28360E-04 0.01438 ];
SM149_CAPT                (idx, [1:   4]) = [  1.55500E+17 0.01286  1.14778E-03 0.01285 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000774 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16254E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000774 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5317152 5.32299E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3679086 3.68307E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1004536 1.00557E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000774 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.69387E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36414E+20 3.9E-06  2.36414E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37699E+19 1.2E-07  9.37699E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35484E+20 0.00030  1.20181E+20 0.00021  1.53031E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29254E+20 0.00018  2.13951E+20 0.00012  1.53031E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.54526E+20 0.00037  2.54526E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.52383E+22 0.00027  3.33923E+22 0.00024  1.84594E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.55953E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.54849E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14903E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28750E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28750E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.65849E-01 0.04944 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33638E-01 0.01289 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.88984E-04 0.02362 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.10019E+03 0.02279 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99446E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.42854E-01 0.03899 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.78145E-01 0.03899 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52122E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.28539E-01 0.00043  5.78517E-02 0.00043  1.84820E-04 0.00942 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.28747E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.28893E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.28747E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03256E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03047E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02908E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32729E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32890E-02 0.00103 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.57621E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.58044E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.51186E-03 0.00563  2.78008E-04 0.01980  8.63336E-04 0.01122  6.38480E-04 0.01386  1.37885E-03 0.00797  2.94917E-04 0.02005  5.82601E-05 0.04162 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.54160E-01 0.01501  1.24761E-02 2.8E-05  3.23279E-02 0.00010  1.06168E-01 0.00061  2.97745E-01 0.00028  1.23715E+00 0.00035  4.77297E+00 0.03863 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22164E-03 0.00818  2.58022E-04 0.02956  7.94967E-04 0.01568  5.82353E-04 0.02028  1.26863E-03 0.01257  2.63542E-04 0.02794  5.41233E-05 0.06347 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.50865E-01 0.02135  1.24762E-02 3.5E-05  3.23326E-02 0.00015  1.06185E-01 0.00087  2.97714E-01 0.00040  1.23720E+00 0.00046  6.18611E+00 0.02832 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.40013E-07 0.00135  9.39710E-07 0.00135  1.03791E-06 0.01722 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.72772E-07 0.00128  8.72491E-07 0.00128  9.63642E-07 0.01721 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18458E-03 0.00933  2.49248E-04 0.03375  7.96311E-04 0.01905  5.84240E-04 0.02160  1.24633E-03 0.01425  2.55625E-04 0.03439  5.28300E-05 0.07265 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.45502E-01 0.02506  1.24753E-02 6.0E-05  3.23352E-02 0.00020  1.06095E-01 0.00100  2.97674E-01 0.00048  1.23717E+00 0.00054  6.17382E+00 0.04150 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.31895E-07 0.01713  8.31479E-07 0.01714  8.77068E-07 0.05736 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.72434E-07 0.01713  7.72048E-07 0.01713  8.14364E-07 0.05742 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80107E-03 0.04288  2.20768E-04 0.14684  6.66232E-04 0.07642  5.68848E-04 0.08679  1.08730E-03 0.06931  1.99751E-04 0.14205  5.81638E-05 0.27236 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.85151E-01 0.10703  1.24768E-02 0.00012  3.23300E-02 0.00049  1.06336E-01 0.00334  2.97153E-01 0.00156  1.23609E+00 0.00214  7.19146E+00 0.11012 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.79887E-03 0.04215  2.13498E-04 0.14221  6.77104E-04 0.07570  5.66907E-04 0.08507  1.08860E-03 0.06871  1.96757E-04 0.14074  5.60047E-05 0.27162 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.69082E-01 0.09994  1.24769E-02 0.00012  3.23297E-02 0.00049  1.06326E-01 0.00332  2.97109E-01 0.00153  1.23606E+00 0.00214  7.20535E+00 0.11005 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.37760E+03 0.03961 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.32431E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.65727E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19268E-03 0.00633 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.42491E+03 0.00639 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89534E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67782E-05 0.00875  6.67688E-05 0.00876  1.21030E-06 0.38316 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.77082E-05 0.02871  5.77646E-05 0.02871  9.78286E-07 0.50377 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79607E-04 0.02298  1.79534E-04 0.02312  1.89240E-04 0.37804 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70166E+01 0.01069 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50943E+01 0.00025  4.07023E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 11:35:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02258E+00  1.00236E+00  1.00285E+00  9.94025E-01  9.97187E-01  1.00226E+00  1.00492E+00  1.00411E+00  9.97044E-01  1.00413E+00  9.90302E-01  9.96502E-01  9.90334E-01  1.00334E+00  9.89629E-01  1.00551E+00  9.97736E-01  9.95208E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.64479E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73552E-01 8.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30195E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44465E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69217E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53263E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52254E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.60669E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.88793E-01 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500761 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50030E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50030E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.17116E+03 ;
RUNNING_TIME              (idx, 1)        =  2.19195E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.67650E-01  9.55667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.17210E+02  3.50564E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.38833E-02  1.52333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.00000E-01  4.88500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.19191E+02  4.95618E+02 ];
CPU_USAGE                 (idx, 1)        = 9.90516 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98459E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91402E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.61611E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66791E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.85607E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.98550E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38168E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.59783E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49696E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27055E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.91718E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.73908E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.60954E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.91976E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.05531E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.05753E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.29216E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.29149E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.48492E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.85282E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.77513E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.11433E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54482E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.10603E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85734E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.03573E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60095E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -6.06450E-03 -1.30492E+27  2.16479E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.92512E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.65882E+18 0.00371  1.76916E-02 0.00371 ];
U233_FISS                 (idx, [1:   4]) = [  9.02109E+19 0.00053  9.62072E-01 0.00011 ];
U235_FISS                 (idx, [1:   4]) = [  8.81653E+17 0.00536  9.40362E-03 0.00539 ];
PU239_FISS                (idx, [1:   4]) = [  5.22315E+13 0.70624  5.51747E-07 0.70623 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00732E+20 0.00054  7.23085E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31682E+19 0.00148  9.45277E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  3.08972E+17 0.00879  2.21816E-03 0.00882 ];
PU239_CAPT                (idx, [1:   4]) = [  5.19396E+13 0.70635  3.72442E-07 0.70639 ];
SM149_CAPT                (idx, [1:   4]) = [  1.69105E+17 0.01239  1.21379E-03 0.01240 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001194 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14296E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001194 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5374436 5.38017E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3617767 3.62134E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1008991 1.00992E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001194 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.06754E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36375E+20 3.7E-06  2.36375E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37621E+19 1.1E-07  9.37621E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.39305E+20 0.00032  1.23724E+20 0.00023  1.55811E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33067E+20 0.00019  2.17486E+20 0.00013  1.55811E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58933E+20 0.00036  2.58933E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.59602E+22 0.00026  3.40884E+22 0.00023  1.87177E+21 0.00186 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61514E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59219E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17230E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27657E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27657E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.44471E-01 0.04905 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33253E-01 0.01314 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.82370E-04 0.02479 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.85055E+03 0.02421 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99010E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.37092E-01 0.03858 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.72723E-01 0.03858 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52101E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99703E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.12934E-01 0.00045  5.68715E-02 0.00044  1.86296E-04 0.00967 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12943E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.12930E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12943E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01551E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02624E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02572E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33293E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33333E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.62600E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.62615E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.63023E-03 0.00591  2.94670E-04 0.01971  8.78475E-04 0.01076  6.65033E-04 0.01315  1.42869E-03 0.00964  3.05890E-04 0.01862  5.74665E-05 0.04620 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47655E-01 0.01478  1.24764E-02 2.6E-05  3.23224E-02 0.00010  1.06169E-01 0.00059  2.98023E-01 0.00028  1.23618E+00 0.00254  4.44658E+00 0.04116 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29131E-03 0.00852  2.63248E-04 0.02795  8.05869E-04 0.01566  5.87583E-04 0.01955  1.30635E-03 0.01371  2.73731E-04 0.03005  5.45246E-05 0.06606 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.52888E-01 0.02156  1.24765E-02 3.3E-05  3.23227E-02 0.00014  1.06145E-01 0.00086  2.97940E-01 0.00037  1.23880E+00 0.00048  6.36226E+00 0.02796 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.59790E-07 0.00127  9.59464E-07 0.00127  1.05388E-06 0.01681 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.76159E-07 0.00120  8.75862E-07 0.00120  9.62047E-07 0.01682 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27164E-03 0.00985  2.69998E-04 0.03191  7.92954E-04 0.01945  6.01979E-04 0.02201  1.28944E-03 0.01550  2.66336E-04 0.03212  5.09291E-05 0.07412 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.52468E-01 0.02651  1.24767E-02 4.1E-05  3.23253E-02 0.00019  1.06070E-01 0.00101  2.98070E-01 0.00047  1.23930E+00 0.00068  6.54357E+00 0.04011 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.44644E-07 0.01714  8.44205E-07 0.01714  9.46735E-07 0.05175 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.71032E-07 0.01713  7.70631E-07 0.01713  8.63914E-07 0.05170 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11248E-03 0.04269  2.11298E-04 0.13951  7.19459E-04 0.08332  6.01189E-04 0.08926  1.25168E-03 0.06573  2.64958E-04 0.13631  6.38980E-05 0.25665 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.85782E-01 0.11073  1.24739E-02 0.00018  3.22997E-02 0.00049  1.06073E-01 0.00319  2.98061E-01 0.00161  1.23767E+00 0.00127  6.22224E+00 0.11547 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09564E-03 0.04196  2.11834E-04 0.13640  7.01557E-04 0.08087  5.91190E-04 0.08707  1.26516E-03 0.06416  2.61633E-04 0.13063  6.42693E-05 0.24938 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.88151E-01 0.10987  1.24738E-02 0.00018  3.23001E-02 0.00049  1.06071E-01 0.00319  2.98042E-01 0.00159  1.23771E+00 0.00127  6.23161E+00 0.11549 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.70935E+03 0.03971 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.51046E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.68173E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34350E-03 0.00572 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.51571E+03 0.00569 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91240E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.57087E-05 0.00959  6.57327E-05 0.00963  7.03292E-07 0.52081 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.66489E-05 0.02804  5.65872E-05 0.02808  6.44418E-07 0.56623 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.72867E-04 0.02404  1.73087E-04 0.02407  1.16923E-04 0.50063 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70314E+01 0.01106 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52254E+01 0.00027  4.07276E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 12:11:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02009E+00  9.91721E-01  1.00005E+00  9.87469E-01  1.00521E+00  1.01492E+00  9.98770E-01  9.98692E-01  1.00094E+00  1.00345E+00  9.96876E-01  1.00323E+00  9.93577E-01  9.96706E-01  9.94243E-01  1.00379E+00  1.00068E+00  9.89598E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.64797E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73520E-01 8.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30002E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44289E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.68768E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54486E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53482E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.63677E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.92075E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500305 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50047E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50047E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.52790E+03 ;
RUNNING_TIME              (idx, 1)        =  2.55037E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.69700E-01  1.02050E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.52914E+02  3.57043E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.08983E-01  1.51000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.53850E-01  5.38000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.55033E+02  5.01248E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91188 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98454E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92127E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.70354E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.68214E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.16206E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02258E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40204E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.64737E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50839E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.76581E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.22057E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.28710E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.89929E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.20613E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.32739E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32642E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.28812E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.28472E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.47770E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.56824E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27682E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.15808E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.65566E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.89673E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.96060E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.05661E+16 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.24072E-02 -2.66970E+27  2.17843E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03248E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.67236E+18 0.00394  1.78283E-02 0.00384 ];
U233_FISS                 (idx, [1:   4]) = [  8.78933E+19 0.00053  9.37079E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.43448E+18 0.00337  2.59548E-02 0.00331 ];
PU239_FISS                (idx, [1:   4]) = [  1.21261E+15 0.16129  1.29610E-05 0.16092 ];
PU240_FISS                (idx, [1:   4]) = [  7.92242E+13 0.57591  8.55940E-07 0.57592 ];
TH232_CAPT                (idx, [1:   4]) = [  1.02172E+20 0.00052  7.08971E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28640E+19 0.00142  8.92630E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  8.63194E+17 0.00547  5.98984E-03 0.00546 ];
U238_CAPT                 (idx, [1:   4]) = [  5.30728E+13 0.70631  3.69192E-07 0.70627 ];
PU239_CAPT                (idx, [1:   4]) = [  3.69717E+14 0.26288  2.57606E-06 0.26288 ];
PU240_CAPT                (idx, [1:   4]) = [  7.95936E+13 0.57594  5.52338E-07 0.57598 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74851E+17 0.01167  1.21334E-03 0.01167 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001896 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14604E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001896 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5450383 5.45572E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3547529 3.55083E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1003984 1.00490E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001896 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.96161E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36271E+20 4.0E-06  2.36271E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37391E+19 1.2E-07  9.37391E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44008E+20 0.00031  1.27946E+20 0.00021  1.60622E+19 0.00182 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37747E+20 0.00019  2.21685E+20 0.00012  1.60622E+19 0.00182 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64153E+20 0.00037  2.64153E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67753E+22 0.00024  3.48433E+22 0.00022  1.93190E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.65462E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64293E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19917E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.24898E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.24898E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.32458E-01 0.04879 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33764E-01 0.01253 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.82274E-04 0.02385 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.96049E+03 0.02216 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99512E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.37056E-01 0.03755 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.72988E-01 0.03755 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52052E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99752E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.95077E-01 0.00045  5.57491E-02 0.00044  1.89397E-04 0.00974 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.95008E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  8.94497E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.95008E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95063E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.99831E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.99869E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.37065E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36984E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.73363E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.72678E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.82673E-03 0.00577  3.03670E-04 0.01992  9.11538E-04 0.01135  7.05730E-04 0.01246  1.51204E-03 0.00923  3.27653E-04 0.01819  6.60978E-05 0.04024 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.70148E-01 0.01507  1.24439E-02 0.00251  3.22980E-02 0.00012  1.06449E-01 0.00063  2.98788E-01 0.00027  1.24424E+00 0.00054  5.09884E+00 0.03598 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.36214E-03 0.00828  2.71910E-04 0.02868  8.09606E-04 0.01727  6.16573E-04 0.01880  1.31550E-03 0.01325  2.93524E-04 0.02692  5.50302E-05 0.05872 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.63352E-01 0.02036  1.24754E-02 3.8E-05  3.22930E-02 0.00015  1.06453E-01 0.00084  2.98763E-01 0.00040  1.24385E+00 0.00066  6.65472E+00 0.02433 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.72702E-07 0.00126  9.72307E-07 0.00127  1.08864E-06 0.01923 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.70562E-07 0.00115  8.70209E-07 0.00116  9.74245E-07 0.01919 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.38426E-03 0.00993  2.58467E-04 0.03299  8.29824E-04 0.02000  6.13722E-04 0.02230  1.33657E-03 0.01494  2.88006E-04 0.03025  5.76730E-05 0.07030 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.73331E-01 0.02517  1.24752E-02 5.8E-05  3.22955E-02 0.00021  1.06428E-01 0.00114  2.98978E-01 0.00048  1.24590E+00 0.00102  6.94421E+00 0.03443 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.55425E-07 0.01708  8.55139E-07 0.01708  8.99120E-07 0.05816 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.65912E-07 0.01708  7.65656E-07 0.01708  8.04653E-07 0.05811 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.19240E-03 0.04386  2.69819E-04 0.14195  7.33277E-04 0.08435  5.78638E-04 0.09250  1.29931E-03 0.06429  2.74821E-04 0.12759  3.65375E-05 0.32191 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.53326E-01 0.08998  1.24784E-02 8.3E-05  3.22691E-02 0.00059  1.07162E-01 0.00369  2.99301E-01 0.00185  1.24626E+00 0.00303  7.62511E+00 0.10176 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.18409E-03 0.04319  2.58742E-04 0.14492  7.37900E-04 0.08313  5.80508E-04 0.09066  1.29752E-03 0.06219  2.74023E-04 0.12577  3.53945E-05 0.32403 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.50180E-01 0.08817  1.24785E-02 8.3E-05  3.22706E-02 0.00057  1.07116E-01 0.00365  2.99260E-01 0.00183  1.24629E+00 0.00302  7.59361E+00 0.10210 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77807E+03 0.04146 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.64400E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.63144E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.45187E-03 0.00680 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.57998E+03 0.00684 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93486E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.62611E-05 0.00945  6.62830E-05 0.00944  3.64551E-07 0.59548 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.56420E-05 0.03043  5.55931E-05 0.03048  5.21857E-07 0.74301 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.72967E-04 0.02290  1.73331E-04 0.02294  8.43255E-05 0.58063 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69167E+01 0.01129 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53482E+01 0.00030  4.05579E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 12:47:15 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01808E+00  9.94115E-01  1.00579E+00  9.89626E-01  1.00542E+00  1.00266E+00  9.97820E-01  1.00456E+00  1.00520E+00  9.99493E-01  9.91632E-01  9.97350E-01  9.95853E-01  1.00770E+00  9.93436E-01  1.00164E+00  9.93161E-01  9.96461E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.65538E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73446E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29827E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44155E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69210E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52801E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51796E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.60977E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.92350E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500307 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50040E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50040E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.88729E+03 ;
RUNNING_TIME              (idx, 1)        =  2.91145E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.68300E-01  9.86000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.88884E+02  3.59702E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.23917E-01  1.49333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.03517E-01  4.96167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.91141E+02  5.06048E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91702 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98410E+00 1.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.92684E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.89651E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73301E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.76225E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.04199E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.41004E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82007E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55768E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.20434E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78272E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.12430E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.31745E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04488E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.84404E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.37841E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37513E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34529E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56393E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.85676E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30248E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27928E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.93262E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.50171E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14819E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.06321E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64364E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.33217E-02 -5.01821E+27  2.20192E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.04330E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.63874E+18 0.00397  1.74848E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  8.43843E+19 0.00054  9.00391E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.84516E+18 0.00224  5.16975E-02 0.00216 ];
PU239_FISS                (idx, [1:   4]) = [  2.60011E+16 0.03050  2.77469E-04 0.03050 ];
PU240_FISS                (idx, [1:   4]) = [  6.12720E+14 0.21231  6.52372E-06 0.21241 ];
PU241_FISS                (idx, [1:   4]) = [  6.38411E+14 0.19817  6.80342E-06 0.19818 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00038E+20 0.00055  6.86816E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23339E+19 0.00141  8.46812E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  1.71612E+18 0.00385  1.17826E-02 0.00385 ];
U238_CAPT                 (idx, [1:   4]) = [  2.13180E+14 0.35052  1.46231E-06 0.35053 ];
PU239_CAPT                (idx, [1:   4]) = [  1.21051E+16 0.04976  8.31606E-05 0.04981 ];
PU240_CAPT                (idx, [1:   4]) = [  2.10112E+15 0.11007  1.44289E-05 0.11011 ];
PU241_CAPT                (idx, [1:   4]) = [  1.60749E+14 0.40571  1.10233E-06 0.40572 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81186E+17 0.01220  1.24416E-03 0.01222 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001593 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12549E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001593 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5474226 5.47981E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3522702 3.52594E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1004665 1.00550E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001593 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.34110E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36147E+20 4.3E-06  2.36147E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36998E+19 1.1E-07  9.36998E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45655E+20 0.00028  1.29541E+20 0.00020  1.61146E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39355E+20 0.00017  2.23240E+20 0.00012  1.61146E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65803E+20 0.00034  2.65803E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.69289E+22 0.00025  3.49961E+22 0.00023  1.93277E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.67273E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66082E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20217E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.17733E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.17733E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.99674E-01 0.05055 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31157E-01 0.01296 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.84805E-04 0.02542 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.97724E+03 0.02305 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99452E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.10292E-01 0.03940 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.48945E-01 0.03941 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52025E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99835E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.88536E-01 0.00046  5.53508E-02 0.00044  1.89323E-04 0.00903 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.88506E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88472E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.88506E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87841E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94139E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94140E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.45102E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.45063E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.86872E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.86530E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93473E-03 0.00599  2.95787E-04 0.02029  9.16144E-04 0.01164  7.25682E-04 0.01245  1.57363E-03 0.00871  3.50998E-04 0.01837  7.24855E-05 0.03727 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.82987E-01 0.01487  1.24451E-02 0.00251  3.22610E-02 0.00013  1.06542E-01 0.00055  3.00062E-01 0.00030  1.25320E+00 0.00070  5.41524E+00 0.03210 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42613E-03 0.00823  2.54888E-04 0.02807  8.00377E-04 0.01668  6.33330E-04 0.01847  1.37664E-03 0.01314  2.98448E-04 0.02556  6.24447E-05 0.05575 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.80789E-01 0.02043  1.24756E-02 5.0E-05  3.22667E-02 0.00018  1.06512E-01 0.00076  3.00097E-01 0.00042  1.25261E+00 0.00090  6.64929E+00 0.02301 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.57011E-07 0.00138  9.56720E-07 0.00137  1.04633E-06 0.01512 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.50265E-07 0.00129  8.50006E-07 0.00129  9.29423E-07 0.01507 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40397E-03 0.00909  2.67926E-04 0.03035  8.02888E-04 0.01857  6.17240E-04 0.02180  1.35099E-03 0.01495  3.02773E-04 0.03107  6.21561E-05 0.06761 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.81682E-01 0.02651  1.24750E-02 5.9E-05  3.22523E-02 0.00021  1.06659E-01 0.00114  3.00174E-01 0.00054  1.25297E+00 0.00125  6.70877E+00 0.03378 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.41380E-07 0.01712  8.40911E-07 0.01712  9.54102E-07 0.05938 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.47697E-07 0.01712  7.47282E-07 0.01712  8.47423E-07 0.05940 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.84573E-03 0.04374  1.85386E-04 0.14537  6.54900E-04 0.07966  4.61218E-04 0.09258  1.17781E-03 0.06879  2.83500E-04 0.12081  8.29152E-05 0.27519 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.48548E-01 0.11836  1.24773E-02 0.00013  3.22620E-02 0.00079  1.06595E-01 0.00331  3.00764E-01 0.00201  1.25436E+00 0.00367  6.19321E+00 0.10278 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.86303E-03 0.04339  1.86259E-04 0.14138  6.61138E-04 0.07926  4.63041E-04 0.09083  1.17630E-03 0.06774  2.92559E-04 0.12294  8.37310E-05 0.26459 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.43137E-01 0.11517  1.24774E-02 0.00013  3.22618E-02 0.00078  1.06598E-01 0.00330  3.00659E-01 0.00200  1.25418E+00 0.00366  6.19321E+00 0.10278 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.41773E+03 0.04095 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.47933E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.42199E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.36396E-03 0.00621 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.55003E+03 0.00630 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91545E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63886E-05 0.00953  6.63690E-05 0.00958  7.62094E-07 0.45854 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.43505E-05 0.02574  5.43204E-05 0.02577  7.94838E-07 0.55023 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.76896E-04 0.02476  1.77095E-04 0.02473  1.22631E-04 0.44668 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63236E+01 0.01133 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51796E+01 0.00027  4.00520E+01 0.00040 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 13:23:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02735E+00  9.87679E-01  9.99433E-01  9.91651E-01  1.00074E+00  9.88116E-01  1.00520E+00  1.00414E+00  1.00184E+00  9.98361E-01  9.88443E-01  1.00156E+00  9.94232E-01  1.00036E+00  1.00308E+00  1.01053E+00  9.95637E-01  1.00165E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.65955E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73405E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29891E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44233E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69100E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52119E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51112E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59376E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.94041E-01 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500468 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50037E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50037E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.24786E+03 ;
RUNNING_TIME              (idx, 1)        =  3.27725E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.68283E-01  9.99833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.25322E+02  3.64375E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.38700E-01  1.47833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.22700E-01  4.19117E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.27350E+02  5.07775E+02 ];
CPU_USAGE                 (idx, 1)        = 9.91034 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98287E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91984E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.93209E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74212E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.13723E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03899E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40756E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85842E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56684E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.19212E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.00131E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.01779E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.52228E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13445E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04122E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81495E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38427E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35090E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57230E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90757E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73275E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29780E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99250E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.94915E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.17557E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.06310E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30373E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.85540E-02 -6.14406E+27  2.21318E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.04062E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.60911E+18 0.00435  1.71849E-02 0.00428 ];
U233_FISS                 (idx, [1:   4]) = [  8.32700E+19 0.00056  8.89353E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.52879E+18 0.00216  5.90509E-02 0.00213 ];
PU239_FISS                (idx, [1:   4]) = [  5.66931E+16 0.02264  6.05142E-04 0.02258 ];
PU240_FISS                (idx, [1:   4]) = [  2.49305E+15 0.10104  2.66722E-05 0.10112 ];
PU241_FISS                (idx, [1:   4]) = [  2.41816E+15 0.10340  2.58200E-05 0.10339 ];
TH232_CAPT                (idx, [1:   4]) = [  9.87638E+19 0.00051  6.78178E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21792E+19 0.00150  8.36300E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  1.95694E+18 0.00360  1.34385E-02 0.00362 ];
U238_CAPT                 (idx, [1:   4]) = [  2.65408E+14 0.34330  1.82266E-06 0.34246 ];
PU239_CAPT                (idx, [1:   4]) = [  2.77481E+16 0.03137  1.90585E-04 0.03139 ];
PU240_CAPT                (idx, [1:   4]) = [  5.87780E+15 0.07320  4.03687E-05 0.07323 ];
PU241_CAPT                (idx, [1:   4]) = [  6.10475E+14 0.21190  4.18784E-06 0.21183 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82903E+17 0.01202  1.25566E-03 0.01196 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001460 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09272E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001460 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5474118 5.47957E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3519785 3.52295E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1007557 1.00841E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001460 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.09896E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36144E+20 4.4E-06  2.36144E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36853E+19 1.1E-07  9.36853E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45699E+20 0.00029  1.29513E+20 0.00019  1.61860E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39384E+20 0.00018  2.23198E+20 0.00011  1.61860E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65774E+20 0.00036  2.65774E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68911E+22 0.00025  3.49542E+22 0.00023  1.93684E+21 0.00191 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.68024E+19 0.00150 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66187E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20017E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.13973E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.13973E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.76790E-01 0.05048 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.34162E-01 0.01278 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.87960E-04 0.02308 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.89814E+03 0.02294 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99162E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.02191E-01 0.04003 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.41525E-01 0.04003 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52061E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99866E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.87674E-01 0.00045  5.53054E-02 0.00044  1.91812E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.88135E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88561E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.88135E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87692E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92121E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.91683E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.48052E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.48679E-02 0.00106 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.90470E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.91521E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94046E-03 0.00560  2.95054E-04 0.01987  9.27319E-04 0.01121  7.04507E-04 0.01268  1.59080E-03 0.00878  3.49932E-04 0.01754  7.28547E-05 0.04246 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.86933E-01 0.01516  1.24146E-02 0.00355  3.22548E-02 0.00013  1.06569E-01 0.00059  3.00747E-01 0.00029  1.25627E+00 0.00071  5.27069E+00 0.03420 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46659E-03 0.00774  2.66453E-04 0.02897  8.18459E-04 0.01592  6.11098E-04 0.01954  1.40280E-03 0.01204  3.06333E-04 0.02610  6.14541E-05 0.05926 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.79710E-01 0.02035  1.24766E-02 8.0E-05  3.22482E-02 0.00019  1.06453E-01 0.00074  3.00743E-01 0.00043  1.25730E+00 0.00102  6.78121E+00 0.02246 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.43984E-07 0.00132  9.43541E-07 0.00132  1.08819E-06 0.02040 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.37876E-07 0.00123  8.37482E-07 0.00123  9.65758E-07 0.02039 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45107E-03 0.00901  2.61629E-04 0.03396  8.22850E-04 0.01950  6.24042E-04 0.02173  1.37756E-03 0.01423  2.99935E-04 0.03107  6.50544E-05 0.06895 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.87305E-01 0.02569  1.24752E-02 6.0E-05  3.22492E-02 0.00023  1.06594E-01 0.00107  3.00793E-01 0.00055  1.25572E+00 0.00134  6.98893E+00 0.03099 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.28906E-07 0.01711  8.28551E-07 0.01711  8.74009E-07 0.05688 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.36008E-07 0.01709  7.35693E-07 0.01709  7.75764E-07 0.05683 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.04911E-03 0.04224  1.95160E-04 0.17018  7.92534E-04 0.08815  5.41970E-04 0.09220  1.19776E-03 0.06959  2.49440E-04 0.12764  7.22396E-05 0.33516 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.61042E-01 0.08434  1.24760E-02 0.00015  3.22819E-02 0.00071  1.06853E-01 0.00349  3.00662E-01 0.00196  1.25726E+00 0.00368  6.26835E+00 0.09661 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.01066E-03 0.04099  1.85038E-04 0.16833  7.77864E-04 0.08432  5.27991E-04 0.08966  1.19579E-03 0.06823  2.51945E-04 0.12477  7.20347E-05 0.31857 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.66386E-01 0.08101  1.24759E-02 0.00015  3.22825E-02 0.00071  1.06816E-01 0.00346  3.00604E-01 0.00194  1.25760E+00 0.00369  6.26835E+00 0.09661 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.69141E+03 0.03888 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.35483E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.30329E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44988E-03 0.00627 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.68853E+03 0.00629 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91007E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.66093E-05 0.00901  6.65910E-05 0.00901  1.07818E-06 0.41247 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.54300E-05 0.02548  5.55002E-05 0.02544  2.87397E-07 0.47536 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.80716E-04 0.02221  1.80832E-04 0.02216  1.49471E-04 0.40895 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65538E+01 0.01130 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51112E+01 0.00029  3.98547E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 14:00:53 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02462E+00  9.92639E-01  9.97590E-01  9.91450E-01  1.00127E+00  9.93122E-01  1.00344E+00  1.00874E+00  1.00303E+00  1.00229E+00  9.88902E-01  1.00235E+00  9.89620E-01  1.00350E+00  9.94148E-01  1.00963E+00  9.97858E-01  9.95807E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.66360E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73364E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29760E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44126E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69089E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51163E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50160E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57970E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.93570E-01 0.00056  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500649 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50055E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50055E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.60422E+03 ;
RUNNING_TIME              (idx, 1)        =  3.64780E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.06933E+00  1.01050E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.62230E+02  3.69083E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.54467E-01  1.57667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.34337E+00  1.42062E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.63404E+02  5.09622E+02 ];
CPU_USAGE                 (idx, 1)        = 9.88054 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98415E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88688E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.93263E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74306E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.61686E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02366E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39701E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87428E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56882E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.30308E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.18368E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.04864E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.69659E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21156E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20556E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.19997E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37733E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34456E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56418E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.93441E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11515E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29075E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01618E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.30910E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.15552E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.06354E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96383E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.38312E-02 -7.27959E+27  2.22453E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.04029E+00 0.00078 ];
TH232_FISS                (idx, [1:   4]) = [  1.60401E+18 0.00424  1.71198E-02 0.00419 ];
U233_FISS                 (idx, [1:   4]) = [  8.24866E+19 0.00059  8.80405E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.03794E+18 0.00209  6.44449E-02 0.00202 ];
U238_FISS                 (idx, [1:   4]) = [  2.76435E+13 1.00000  2.93427E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  9.61678E+16 0.01623  1.02634E-03 0.01623 ];
PU240_FISS                (idx, [1:   4]) = [  4.28598E+15 0.08264  4.57519E-05 0.08266 ];
PU241_FISS                (idx, [1:   4]) = [  4.92144E+15 0.07366  5.24962E-05 0.07366 ];
TH232_CAPT                (idx, [1:   4]) = [  9.80007E+19 0.00057  6.72172E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20155E+19 0.00144  8.24168E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  2.13092E+18 0.00359  1.46165E-02 0.00360 ];
U238_CAPT                 (idx, [1:   4]) = [  4.77047E+14 0.23065  3.27834E-06 0.23064 ];
PU239_CAPT                (idx, [1:   4]) = [  4.70071E+16 0.02390  3.22383E-04 0.02388 ];
PU240_CAPT                (idx, [1:   4]) = [  1.38093E+16 0.04532  9.46936E-05 0.04527 ];
PU241_CAPT                (idx, [1:   4]) = [  1.16939E+15 0.14607  8.00900E-06 0.14604 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83508E+17 0.01303  1.25888E-03 0.01305 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002196 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10369E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002196 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5478463 5.48352E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3520818 3.52380E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1002915 1.00371E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002196 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.93715E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36163E+20 4.4E-06  2.36163E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36726E+19 1.1E-07  9.36726E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45747E+20 0.00029  1.29586E+20 0.00021  1.61609E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39419E+20 0.00018  2.23258E+20 0.00012  1.61609E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65884E+20 0.00035  2.65884E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68654E+22 0.00025  3.49313E+22 0.00023  1.93411E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.66884E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66108E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19814E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  8.10314E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.10314E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.73478E-01 0.04947 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.51646E-01 0.01129 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.98164E-04 0.02316 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.84256E+03 0.02242 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99631E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.44660E-01 0.03654 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.80142E-01 0.03654 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52115E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99893E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.88326E-01 0.00047  5.53320E-02 0.00046  1.94305E-04 0.00792 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.88475E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88261E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.88475E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87633E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.89919E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90011E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.51350E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.51180E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.95922E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.95388E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.01204E-03 0.00519  2.94482E-04 0.01977  9.52971E-04 0.01056  7.19761E-04 0.01293  1.60169E-03 0.00827  3.69593E-04 0.01782  7.35450E-05 0.03756 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.89989E-01 0.01325  1.24146E-02 0.00355  3.22377E-02 0.00014  1.06679E-01 0.00054  3.00785E-01 0.00031  1.25744E+00 0.00075  5.75863E+00 0.03037 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45310E-03 0.00741  2.43281E-04 0.02975  8.35039E-04 0.01558  6.14090E-04 0.01840  1.38092E-03 0.01167  3.22844E-04 0.02521  5.69263E-05 0.05197 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.80928E-01 0.01760  1.24767E-02 4.4E-05  3.22362E-02 0.00018  1.06716E-01 0.00079  3.00781E-01 0.00044  1.25775E+00 0.00106  7.04594E+00 0.02057 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.37657E-07 0.00128  9.37250E-07 0.00128  1.04349E-06 0.01801 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32872E-07 0.00119  8.32510E-07 0.00119  9.26897E-07 0.01804 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.51111E-03 0.00806  2.58811E-04 0.03322  8.45668E-04 0.01732  6.21044E-04 0.02073  1.39058E-03 0.01401  3.30441E-04 0.02924  6.45713E-05 0.06224 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.87331E-01 0.02237  1.24759E-02 6.5E-05  3.22182E-02 0.00026  1.06776E-01 0.00119  3.00881E-01 0.00055  1.25998E+00 0.00137  6.69457E+00 0.03194 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.27906E-07 0.01726  8.27524E-07 0.01726  9.37780E-07 0.06455 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.35654E-07 0.01724  7.35315E-07 0.01724  8.33407E-07 0.06464 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.46832E-03 0.04247  2.84873E-04 0.14040  7.30351E-04 0.07735  6.35792E-04 0.08573  1.43577E-03 0.06256  3.34163E-04 0.12209  4.73704E-05 0.31379 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.81460E-01 0.09600  1.24743E-02 0.00017  3.22287E-02 0.00078  1.07474E-01 0.00387  3.00565E-01 0.00184  1.26372E+00 0.00400  7.05274E+00 0.08482 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.46789E-03 0.04189  2.79451E-04 0.14077  7.26560E-04 0.07579  6.33102E-04 0.08495  1.44484E-03 0.06181  3.35847E-04 0.12243  4.80917E-05 0.31304 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.86221E-01 0.09626  1.24744E-02 0.00017  3.22292E-02 0.00077  1.07454E-01 0.00383  3.00657E-01 0.00183  1.26364E+00 0.00397  7.05274E+00 0.08482 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.22689E+03 0.03913 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.30051E-07 0.00080 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.26118E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.57117E-03 0.00573 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.84043E+03 0.00573 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91672E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74156E-05 0.00780  6.74049E-05 0.00779  7.38960E-07 0.51020 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.79070E-05 0.02734  5.79785E-05 0.02734  2.59311E-07 0.57436 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.88581E-04 0.02250  1.88935E-04 0.02252  9.99761E-05 0.49981 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64627E+01 0.01193 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50160E+01 0.00028  3.96424E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 14:38:23 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01911E+00  9.89995E-01  1.00481E+00  9.88152E-01  1.00009E+00  9.97351E-01  1.00375E+00  1.00329E+00  1.01222E+00  9.97998E-01  9.90197E-01  1.00397E+00  9.93144E-01  9.99128E-01  9.98605E-01  1.00030E+00  1.00760E+00  9.90289E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.68457E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.73154E-01 8.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29334E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.43809E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69382E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50156E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49150E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57249E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  8.01254E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500832 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50055E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50055E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.95450E+03 ;
RUNNING_TIME              (idx, 1)        =  4.02278E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.17355E+00  1.04217E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.99575E+02  3.73451E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.69583E-01  1.51167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.93335E+00  2.58992E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.99733E+02  5.10816E+02 ];
CPU_USAGE                 (idx, 1)        = 9.83028 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98480E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.83106E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.94983E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74853E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.94066E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.98509E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37393E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.93005E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57662E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.79096E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.76199E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.29998E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04917E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44224E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70321E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.33962E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37242E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33818E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55696E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.97879E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.26956E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28378E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.06693E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.43162E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.12419E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.06598E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60421E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.48246E-02 -1.17968E+28  2.26971E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03782E+00 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.56002E+18 0.00441  1.66769E-02 0.00433 ];
U233_FISS                 (idx, [1:   4]) = [  8.04344E+19 0.00057  8.59926E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.90567E+18 0.00197  7.38255E-02 0.00185 ];
U238_FISS                 (idx, [1:   4]) = [  7.92223E+13 0.57596  8.54710E-07 0.57593 ];
PU239_FISS                (idx, [1:   4]) = [  3.32949E+17 0.00882  3.55939E-03 0.00879 ];
PU240_FISS                (idx, [1:   4]) = [  2.56022E+16 0.03354  2.73846E-04 0.03364 ];
PU241_FISS                (idx, [1:   4]) = [  3.67086E+16 0.02614  3.92372E-04 0.02612 ];
TH232_CAPT                (idx, [1:   4]) = [  9.57746E+19 0.00054  6.54181E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17285E+19 0.00153  8.01109E-02 0.00148 ];
U235_CAPT                 (idx, [1:   4]) = [  2.44099E+18 0.00319  1.66736E-02 0.00319 ];
U238_CAPT                 (idx, [1:   4]) = [  1.23153E+15 0.15782  8.39615E-06 0.15765 ];
PU239_CAPT                (idx, [1:   4]) = [  1.50906E+17 0.01269  1.03072E-03 0.01269 ];
PU240_CAPT                (idx, [1:   4]) = [  6.68605E+16 0.01950  4.56625E-04 0.01947 ];
PU241_CAPT                (idx, [1:   4]) = [  7.19150E+15 0.06088  4.91606E-05 0.06092 ];
SM149_CAPT                (idx, [1:   4]) = [  1.87440E+17 0.01226  1.28046E-03 0.01228 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002198 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09986E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002198 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5488719 5.49374E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3506868 3.50991E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1006611 1.00736E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002198 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.02445E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36334E+20 4.7E-06  2.36334E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36331E+19 1.0E-07  9.36331E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46401E+20 0.00029  1.30009E+20 0.00020  1.63921E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40035E+20 0.00018  2.23642E+20 0.00012  1.63921E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66495E+20 0.00035  2.66495E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.69828E+22 0.00025  3.50334E+22 0.00022  1.94930E+21 0.00176 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.68468E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66881E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19820E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  7.96489E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.96489E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.79431E-01 0.04511 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.40375E-01 0.01228 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.19416E-04 0.02341 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.84355E+03 0.02219 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99268E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.92259E-01 0.03258 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.22610E-01 0.03258 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52404E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99978E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.85879E-01 0.00046  5.51795E-02 0.00046  1.92895E-04 0.00859 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.86523E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.86863E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.86523E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.85793E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.86027E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.85727E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.57363E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.57795E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.05119E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.04999E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03557E-03 0.00577  2.91079E-04 0.01948  9.42166E-04 0.01149  7.39312E-04 0.01232  1.60943E-03 0.00901  3.70200E-04 0.01761  8.33810E-05 0.03537 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13918E-01 0.01465  1.24773E-02 7.4E-05  3.22202E-02 0.00015  1.06879E-01 0.00058  3.01058E-01 0.00030  1.25973E+00 0.00093  6.18870E+00 0.02754 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51776E-03 0.00747  2.49614E-04 0.02986  8.16215E-04 0.01604  6.44571E-04 0.01719  1.40237E-03 0.01287  3.35924E-04 0.02526  6.90666E-05 0.05387 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09784E-01 0.02018  1.24781E-02 0.00013  3.22131E-02 0.00023  1.06880E-01 0.00083  3.00961E-01 0.00043  1.26013E+00 0.00113  7.28531E+00 0.01945 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.24409E-07 0.00136  9.24044E-07 0.00136  1.03508E-06 0.01673 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.18819E-07 0.00122  8.18496E-07 0.00122  9.16700E-07 0.01669 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48696E-03 0.00856  2.48106E-04 0.03283  8.24752E-04 0.01919  6.35899E-04 0.02089  1.39457E-03 0.01445  3.12206E-04 0.02922  7.14239E-05 0.06527 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08504E-01 0.02569  1.24797E-02 0.00023  3.22141E-02 0.00028  1.06898E-01 0.00105  3.01150E-01 0.00055  1.26197E+00 0.00184  7.29653E+00 0.02800 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.14278E-07 0.01717  8.13985E-07 0.01717  8.28894E-07 0.05662 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21255E-07 0.01716  7.20994E-07 0.01716  7.34317E-07 0.05658 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06139E-03 0.04096  2.28080E-04 0.13493  7.88955E-04 0.07891  5.83193E-04 0.09108  1.15519E-03 0.06379  2.70519E-04 0.13901  3.54540E-05 0.34889 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42749E-01 0.08953  1.24757E-02 0.00018  3.22122E-02 0.00068  1.06860E-01 0.00319  3.00615E-01 0.00194  1.26510E+00 0.00419  6.88770E+00 0.12057 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.04139E-03 0.04064  2.15534E-04 0.13256  7.81961E-04 0.07839  5.83961E-04 0.08957  1.16061E-03 0.06286  2.59817E-04 0.13693  3.95103E-05 0.34191 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.47039E-01 0.09376  1.24757E-02 0.00018  3.22102E-02 0.00067  1.06868E-01 0.00319  3.00605E-01 0.00193  1.26540E+00 0.00419  6.85487E+00 0.12150 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.78280E+03 0.03781 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.15101E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10592E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47761E-03 0.00654 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.79992E+03 0.00646 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95495E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71859E-05 0.00764  6.72333E-05 0.00767  1.93031E-06 0.29019 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.48159E-05 0.02546  5.46645E-05 0.02538  2.32885E-06 0.44910 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.11115E-04 0.02232  2.10765E-04 0.02244  2.85075E-04 0.28738 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60815E+01 0.01034 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49150E+01 0.00027  3.92921E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 15:16:00 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01863E+00  9.96253E-01  1.00309E+00  9.97174E-01  9.99774E-01  9.99813E-01  1.00071E+00  1.00399E+00  1.00711E+00  9.93998E-01  9.99565E-01  9.92679E-01  9.91607E-01  1.00514E+00  9.88164E-01  1.00857E+00  1.00222E+00  9.91522E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.73372E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72663E-01 8.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.27846E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.42588E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.70369E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49896E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48882E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61392E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  8.23375E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500426 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50033E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50033E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.30274E+03 ;
RUNNING_TIME              (idx, 1)        =  4.39903E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.27817E+00  1.04617E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.37044E+02  3.74685E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.85450E-01  1.58667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.90200E+00  2.96858E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.36979E+02  5.11910E+02 ];
CPU_USAGE                 (idx, 1)        = 9.78111 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98573E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77636E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.92952E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74288E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.89034E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93410E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34596E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96071E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57375E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.49247E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38881E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.31764E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13747E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69301E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.24043E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57998E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34786E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31853E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53059E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01443E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59897E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24546E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04523E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.12182E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.04342E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.06879E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32052E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.09330E-01 -2.35250E+28  2.38699E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03024E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.54126E+18 0.00443  1.64742E-02 0.00438 ];
U233_FISS                 (idx, [1:   4]) = [  7.92494E+19 0.00056  8.47122E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.11914E+18 0.00189  7.60982E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  1.84971E+14 0.37513  1.96988E-06 0.37515 ];
PU239_FISS                (idx, [1:   4]) = [  6.18553E+17 0.00669  6.61182E-03 0.00666 ];
PU240_FISS                (idx, [1:   4]) = [  7.36100E+16 0.01786  7.87048E-04 0.01787 ];
PU241_FISS                (idx, [1:   4]) = [  1.30861E+17 0.01430  1.39859E-03 0.01426 ];
TH232_CAPT                (idx, [1:   4]) = [  9.37545E+19 0.00054  6.38597E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15408E+19 0.00144  7.86106E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49879E+18 0.00343  1.70195E-02 0.00338 ];
U238_CAPT                 (idx, [1:   4]) = [  2.27310E+15 0.10987  1.54915E-05 0.10983 ];
PU239_CAPT                (idx, [1:   4]) = [  2.84653E+17 0.00915  1.93908E-03 0.00916 ];
PU240_CAPT                (idx, [1:   4]) = [  1.92602E+17 0.01181  1.31176E-03 0.01178 ];
PU241_CAPT                (idx, [1:   4]) = [  2.90407E+16 0.03125  1.97793E-04 0.03125 ];
SM149_CAPT                (idx, [1:   4]) = [  1.97401E+17 0.01140  1.34458E-03 0.01140 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001311 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08928E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001311 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5489161 5.49460E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3498056 3.50123E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1014094 1.01506E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001311 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36628E+20 4.9E-06  2.36628E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35951E+19 1.1E-07  9.35951E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46782E+20 0.00027  1.29844E+20 0.00018  1.69386E+19 0.00169 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40377E+20 0.00016  2.23439E+20 0.00010  1.69386E+19 0.00169 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.67198E+20 0.00036  2.67198E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.73299E+22 0.00025  3.53472E+22 0.00022  1.98264E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.71242E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67501E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20065E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  7.69319E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.69319E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.17659E-01 0.04010 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.11165E-01 0.01039 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.16306E-04 0.01880 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.86339E+03 0.02576 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.98499E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 8.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.41296E-01 0.02871 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.66120E-01 0.02871 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52821E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00059E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.85144E-01 0.00042  5.51317E-02 0.00043  1.89527E-04 0.00935 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.85551E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.85635E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.85551E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.85591E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83305E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83155E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61697E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61908E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10362E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11446E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.00125E-03 0.00583  2.91633E-04 0.02208  9.20840E-04 0.01205  7.20290E-04 0.01270  1.62084E-03 0.00826  3.71104E-04 0.01750  7.65522E-05 0.03810 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.99339E-01 0.01415  1.24479E-02 0.00251  3.22045E-02 0.00016  1.06674E-01 0.00056  3.01412E-01 0.00031  1.25900E+00 0.00107  5.71731E+00 0.03013 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48237E-03 0.00860  2.56133E-04 0.03116  7.94245E-04 0.01685  6.45995E-04 0.01904  1.39835E-03 0.01259  3.23232E-04 0.02514  6.44143E-05 0.05608 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.95102E-01 0.02002  1.24815E-02 0.00020  3.22029E-02 0.00022  1.06690E-01 0.00081  3.01357E-01 0.00044  1.25683E+00 0.00147  6.97412E+00 0.02038 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.20735E-07 0.00144  9.20195E-07 0.00144  1.08207E-06 0.03480 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14921E-07 0.00137  8.14444E-07 0.00137  9.57494E-07 0.03471 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42951E-03 0.00949  2.53655E-04 0.03719  7.88352E-04 0.01949  6.14185E-04 0.02236  1.38689E-03 0.01425  3.23840E-04 0.03077  6.25844E-05 0.06742 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.05708E-01 0.02806  1.24813E-02 0.00028  3.22122E-02 0.00033  1.06401E-01 0.00092  3.01224E-01 0.00055  1.25531E+00 0.00247  7.39721E+00 0.02966 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.08606E-07 0.01728  8.07930E-07 0.01729  8.73830E-07 0.05392 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.15872E-07 0.01729  7.15272E-07 0.01729  7.74033E-07 0.05400 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.30701E-03 0.04368  2.38658E-04 0.15095  6.81047E-04 0.08289  5.69276E-04 0.08907  1.39427E-03 0.06647  3.37645E-04 0.12782  8.61145E-05 0.32236 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.14933E-01 0.09944  1.24771E-02 0.00013  3.21977E-02 0.00085  1.06325E-01 0.00284  3.01221E-01 0.00197  1.25164E+00 0.00610  7.36762E+00 0.08433 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.30774E-03 0.04321  2.44053E-04 0.14481  6.99130E-04 0.08121  5.71732E-04 0.08851  1.38380E-03 0.06736  3.28811E-04 0.12520  8.02171E-05 0.31327 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.04177E-01 0.09312  1.24772E-02 0.00012  3.21965E-02 0.00083  1.06345E-01 0.00284  3.01166E-01 0.00196  1.25155E+00 0.00608  7.36861E+00 0.08429 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.12762E+03 0.04108 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.10465E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.05833E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.45573E-03 0.00577 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.79687E+03 0.00588 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.16010E-09 0.00095 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.85496E-05 0.00549  6.85547E-05 0.00549  2.00185E-06 0.29113 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.91431E-05 0.01986  5.91223E-05 0.01982  1.97191E-06 0.34736 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.03211E-04 0.01836  3.03113E-04 0.01839  3.38313E-04 0.29332 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61780E+01 0.01250 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48882E+01 0.00029  3.90032E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 15:53:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01522E+00  9.91314E-01  9.93705E-01  1.01027E+00  9.94509E-01  1.00030E+00  1.00467E+00  9.95476E-01  9.92542E-01  9.99996E-01  9.99003E-01  9.99572E-01  9.99232E-01  1.00165E+00  9.98134E-01  9.97912E-01  1.00875E+00  9.97742E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.79302E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.72070E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.26299E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.41362E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71963E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50684E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49657E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.67593E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  8.51332E-01 0.00054  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500638 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50044E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50044E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.65151E+03 ;
RUNNING_TIME              (idx, 1)        =  4.77660E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.38255E+00  1.04383E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.74643E+02  3.75992E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.01133E-01  1.56833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.09586E+01  3.05650E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.74649E+02  5.12238E+02 ];
CPU_USAGE                 (idx, 1)        = 9.73811 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98585E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.72846E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.92245E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74099E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.37412E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91790E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33660E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96975E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57269E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57393E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55833E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39155E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14843E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76569E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39847E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93315E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34135E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31378E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52392E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.03103E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.01065E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23209E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03482E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27860E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01521E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.06992E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98061E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.65031E-01 -3.55104E+28  2.50684E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02171E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.53118E+18 0.00428  1.63560E-02 0.00420 ];
U233_FISS                 (idx, [1:   4]) = [  7.92121E+19 0.00058  8.46198E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.08287E+18 0.00188  7.56644E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  1.60501E+14 0.40569  1.71795E-06 0.40571 ];
PU239_FISS                (idx, [1:   4]) = [  6.54504E+17 0.00627  6.99186E-03 0.00624 ];
PU240_FISS                (idx, [1:   4]) = [  8.53150E+16 0.01732  9.11404E-04 0.01733 ];
PU241_FISS                (idx, [1:   4]) = [  1.52634E+17 0.01328  1.63050E-03 0.01328 ];
TH232_CAPT                (idx, [1:   4]) = [  9.28376E+19 0.00054  6.32964E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15189E+19 0.00151  7.85382E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48588E+18 0.00311  1.69497E-02 0.00314 ];
U238_CAPT                 (idx, [1:   4]) = [  2.86346E+15 0.09364  1.95052E-05 0.09367 ];
PU239_CAPT                (idx, [1:   4]) = [  3.05449E+17 0.00887  2.08234E-03 0.00884 ];
PU240_CAPT                (idx, [1:   4]) = [  2.27400E+17 0.01103  1.55049E-03 0.01103 ];
PU241_CAPT                (idx, [1:   4]) = [  3.17455E+16 0.02846  2.16390E-04 0.02842 ];
SM149_CAPT                (idx, [1:   4]) = [  2.00837E+17 0.01130  1.36935E-03 0.01131 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001754 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07142E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001754 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5478393 5.48350E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3496694 3.49970E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1026667 1.02751E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001754 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.02445E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36682E+20 4.7E-06  2.36682E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35900E+19 1.1E-07  9.35900E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46752E+20 0.00029  1.29257E+20 0.00019  1.74948E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40342E+20 0.00018  2.22847E+20 0.00011  1.74948E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.67480E+20 0.00034  2.67480E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.77020E+22 0.00024  3.56910E+22 0.00022  2.01104E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.74850E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67827E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20397E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  7.47220E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.47220E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.98750E-01 0.03450 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.08916E-01 0.00836 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.10418E-04 0.01642 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.47085E+03 0.02733 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.97257E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99991E-01 9.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.33137E-01 0.02145 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.47519E-01 0.02145 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52893E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.85065E-01 0.00045  5.51238E-02 0.00044  1.90838E-04 0.00886 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.84678E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.84902E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.84678E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.85967E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83238E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83244E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61812E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61761E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11761E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.12212E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03720E-03 0.00510  2.89220E-04 0.02124  9.46227E-04 0.01054  7.31749E-04 0.01176  1.61717E-03 0.00815  3.70996E-04 0.01857  8.18377E-05 0.04039 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08666E-01 0.01545  1.24475E-02 0.00251  3.21945E-02 0.00016  1.06842E-01 0.00057  3.01361E-01 0.00031  1.25740E+00 0.00104  5.77043E+00 0.03074 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46981E-03 0.00774  2.38986E-04 0.03061  8.05983E-04 0.01650  6.37124E-04 0.01745  1.38952E-03 0.01226  3.24872E-04 0.02604  7.33271E-05 0.05503 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.18085E-01 0.02165  1.24784E-02 5.9E-05  3.21965E-02 0.00022  1.06990E-01 0.00081  3.01543E-01 0.00044  1.25824E+00 0.00133  7.35084E+00 0.01812 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.33526E-07 0.00160  9.33043E-07 0.00160  1.07067E-06 0.02325 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.26137E-07 0.00147  8.25709E-07 0.00147  9.47611E-07 0.02325 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45601E-03 0.00889  2.48001E-04 0.03374  8.15741E-04 0.01842  6.24839E-04 0.02048  1.38090E-03 0.01447  3.17937E-04 0.03033  6.85922E-05 0.06481 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.13141E-01 0.02522  1.24782E-02 0.00012  3.21928E-02 0.00029  1.06843E-01 0.00103  3.01291E-01 0.00053  1.26108E+00 0.00170  7.52561E+00 0.02704 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.12332E-07 0.01727  8.11922E-07 0.01727  8.35434E-07 0.05583 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.18956E-07 0.01726  7.18595E-07 0.01726  7.38832E-07 0.05577 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11038E-03 0.04478  2.25565E-04 0.15205  7.25140E-04 0.08379  6.02830E-04 0.08811  1.25844E-03 0.06605  2.19554E-04 0.14571  7.88567E-05 0.25340 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.08861E-01 0.12984  1.24752E-02 0.00019  3.21390E-02 0.00096  1.07104E-01 0.00330  3.01977E-01 0.00215  1.26362E+00 0.00650  7.99094E+00 0.06481 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09351E-03 0.04371  2.22228E-04 0.15072  7.22057E-04 0.08354  6.07587E-04 0.08674  1.23908E-03 0.06463  2.22198E-04 0.14322  8.03546E-05 0.24823 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  5.14807E-01 0.12952  1.24752E-02 0.00019  3.21405E-02 0.00095  1.07052E-01 0.00324  3.01939E-01 0.00213  1.26393E+00 0.00650  7.99094E+00 0.06481 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.85641E+03 0.04184 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.22439E-07 0.00085 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.16357E-07 0.00074 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44016E-03 0.00686 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.72906E+03 0.00675 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.39756E-09 0.00099 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.90198E-05 0.00453  6.90210E-05 0.00451  1.61724E-06 0.33515 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.94071E-05 0.01616  5.92959E-05 0.01614  2.34359E-06 0.42099 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.92629E-04 0.01573  3.93212E-04 0.01576  2.46551E-04 0.31539 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60976E+01 0.01111 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49657E+01 0.00028  3.89883E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha6' ;
HOSTNAME                  (idx, [1: 24])  = 'coragem.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Wed Apr 20 07:56:06 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 20 16:14:21 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650452166571 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03291E+00  1.00611E+00  9.99378E-01  9.96790E-01  9.98855E-01  9.93896E-01  1.00331E+00  9.98156E-01  9.93190E-01  9.93112E-01  9.98737E-01  9.85140E-01  9.98313E-01  9.95928E-01  1.00234E+00  1.00535E+00  9.97150E-01  1.00133E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.84878E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.71512E-01 9.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.25005E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.40360E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73080E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51640E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50604E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.73306E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  8.79041E-01 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 2500061 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  2.50048E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  2.50048E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.83942E+03 ;
RUNNING_TIME              (idx, 1)        =  4.98257E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.14720E+00  1.14720E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.48758E+00  1.05033E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.95079E+02  2.04360E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.16283E-01  1.51500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.05833E-02  1.05833E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.28263E+01  1.86768E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.96435E+02  4.96435E+02 ];
CPU_USAGE                 (idx, 1)        = 9.71270 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97695E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.70066E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.90 ;
ALLOC_MEMSIZE             (idx, 1)        = 22569.04;
MEMSIZE                   (idx, 1)        = 22361.43;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 209.72;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 207.62;

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

TOT_ACTIVITY              (idx, 1)        =  6.91590E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74041E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.54967E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90718E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32984E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.97389E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57274E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.58359E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60434E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39896E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14642E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78725E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44631E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03532E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34056E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31330E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52315E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.03524E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13825E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22434E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03251E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.30274E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99496E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.07136E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64071E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.21165E-01 -4.75890E+28  2.62763E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01644E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.53824E+18 0.00421  1.64377E-02 0.00414 ];
U233_FISS                 (idx, [1:   4]) = [  7.91722E+19 0.00056  8.46102E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.06043E+18 0.00204  7.54512E-02 0.00193 ];
U238_FISS                 (idx, [1:   4]) = [  2.96635E+14 0.32399  3.18200E-06 0.32440 ];
PU239_FISS                (idx, [1:   4]) = [  6.59818E+17 0.00657  7.05168E-03 0.00656 ];
PU240_FISS                (idx, [1:   4]) = [  8.58397E+16 0.01774  9.17338E-04 0.01772 ];
PU241_FISS                (idx, [1:   4]) = [  1.55170E+17 0.01349  1.65838E-03 0.01350 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22312E+19 0.00053  6.28312E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15106E+19 0.00156  7.84148E-02 0.00151 ];
U235_CAPT                 (idx, [1:   4]) = [  2.46946E+18 0.00321  1.68235E-02 0.00321 ];
U238_CAPT                 (idx, [1:   4]) = [  3.21745E+15 0.09221  2.19382E-05 0.09224 ];
PU239_CAPT                (idx, [1:   4]) = [  3.07167E+17 0.00945  2.09278E-03 0.00947 ];
PU240_CAPT                (idx, [1:   4]) = [  2.28177E+17 0.01056  1.55442E-03 0.01055 ];
PU241_CAPT                (idx, [1:   4]) = [  3.39483E+16 0.02796  2.31265E-04 0.02800 ];
SM149_CAPT                (idx, [1:   4]) = [  2.03292E+17 0.01042  1.38493E-03 0.01041 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001903 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07662E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001903 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5475569 5.48068E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3490660 3.49366E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1035674 1.03643E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001903 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.84402E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61696E-02 2.7E-09  3.61696E-02 2.7E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36691E+20 4.8E-06  2.36691E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35897E+19 1.1E-07  9.35897E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46815E+20 0.00028  1.28644E+20 0.00019  1.81714E+19 0.00162 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40405E+20 0.00017  2.22233E+20 0.00011  1.81714E+19 0.00162 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.67839E+20 0.00035  2.67839E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.80705E+22 0.00026  3.60237E+22 0.00024  2.04677E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.77611E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.68166E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20813E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29426E+04 ;
TOT_FMASS                 (idx, 1)        =  7.28913E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29426E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.28913E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.40471E-01 0.02943 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.09559E-01 0.00731 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.22313E-04 0.01515 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.62931E+03 0.02987 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.96366E-01 0.00016 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.11766E-01 0.01426 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.17255E-01 0.01426 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52903E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.83400E-01 0.00046  5.50286E-02 0.00044  1.92593E-04 0.00942 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.83597E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.83747E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.83597E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.85749E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83580E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83431E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61251E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61458E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.12242E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.12359E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.05816E-03 0.00552  2.85272E-04 0.02042  9.54746E-04 0.01115  7.34056E-04 0.01198  1.62108E-03 0.00837  3.76354E-04 0.01675  8.66486E-05 0.03349 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13180E-01 0.01378  1.24171E-02 0.00355  3.22017E-02 0.00015  1.06870E-01 0.00059  3.01358E-01 0.00032  1.25751E+00 0.00104  6.00850E+00 0.02661 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47704E-03 0.00807  2.43060E-04 0.02971  8.25400E-04 0.01619  6.35724E-04 0.01765  1.38296E-03 0.01266  3.15049E-04 0.02409  7.48508E-05 0.05053 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09534E-01 0.01874  1.24801E-02 0.00014  3.22014E-02 0.00023  1.07034E-01 0.00087  3.01307E-01 0.00044  1.25855E+00 0.00152  6.93147E+00 0.01989 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.52900E-07 0.00166  9.52532E-07 0.00167  1.05531E-06 0.01869 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.41714E-07 0.00158  8.41389E-07 0.00159  9.32210E-07 0.01869 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48067E-03 0.00950  2.39617E-04 0.03465  8.25561E-04 0.01911  6.11225E-04 0.02190  1.40879E-03 0.01387  3.28456E-04 0.02961  6.70241E-05 0.06284 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.05514E-01 0.02397  1.24767E-02 7.4E-05  3.22185E-02 0.00027  1.06703E-01 0.00103  3.01396E-01 0.00054  1.25671E+00 0.00198  7.29989E+00 0.02876 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.23834E-07 0.01740  8.23562E-07 0.01741  9.22410E-07 0.05328 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.27692E-07 0.01737  7.27452E-07 0.01738  8.14547E-07 0.05321 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.27648E-03 0.04323  2.21616E-04 0.13412  7.99170E-04 0.07929  5.79009E-04 0.09640  1.32699E-03 0.06679  3.00956E-04 0.12178  4.87424E-05 0.29042 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.94679E-01 0.10814  1.24765E-02 0.00013  3.22111E-02 0.00086  1.06862E-01 0.00319  3.01631E-01 0.00189  1.24970E+00 0.00521  7.24055E+00 0.09404 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.29069E-03 0.04262  2.26041E-04 0.13105  7.96888E-04 0.07816  5.74315E-04 0.09541  1.32996E-03 0.06648  3.14086E-04 0.11846  4.94085E-05 0.29238 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.96555E-01 0.10765  1.24765E-02 0.00013  3.22107E-02 0.00086  1.06825E-01 0.00317  3.01576E-01 0.00187  1.24995E+00 0.00522  7.24055E+00 0.09404 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.99890E+03 0.03976 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.37351E-07 0.00087 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.27980E-07 0.00071 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53480E-03 0.00647 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77166E+03 0.00647 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.65626E-09 0.00108 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.98095E-05 0.00412  6.97911E-05 0.00414  3.22551E-06 0.24833 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.23037E-05 0.01394  6.22639E-05 0.01389  2.72138E-06 0.39897 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.98174E-04 0.01463  4.98396E-04 0.01459  4.60788E-04 0.23322 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60815E+01 0.01084 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50604E+01 0.00027  3.90317E+01 0.00038 ];

