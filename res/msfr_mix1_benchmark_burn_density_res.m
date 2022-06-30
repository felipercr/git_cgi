
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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 10:34:47 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.31725E+00  1.03555E+00  9.68675E-01  9.87575E-01  1.00471E+00  9.74289E-01  9.77731E-01  9.66823E-01  9.62581E-01  9.79000E-01  9.59414E-01  9.84202E-01  9.47717E-01  9.68675E-01  9.78383E-01  9.82419E-01  9.84568E-01  1.02042E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.27443E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77256E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.08422E-01 0.00015  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.23831E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.78492E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.87054E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.87054E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.01700E+02 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.12140E-02 0.00164  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428652 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42890E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42890E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.44717E+00 ;
RUNNING_TIME              (idx, 1)        =  1.28948E+00 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.17500E-02  3.17500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.96117E-01  6.96117E-01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  9.60500E-02  2.53333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.28038E+00  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 5.77531 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.80887E+00 0.00028 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  5.39997E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  2.22071E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74640E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.11243E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.22071E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.74640E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.13501E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.13312E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.13501E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13312E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.28100E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.11146E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.25165E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.66616E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15185E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.09555E+00 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.64283E+18 0.00383  1.75210E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  9.21116E+19 0.00050  9.82479E-01 6.7E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  1.16058E+20 0.00050  8.02887E-01 0.00022 ];
U233_CAPT                 (idx, [1:   4]) = [  1.38315E+19 0.00127  9.56902E-02 0.00123 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002282 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.20483E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002282 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6066951 6.07297E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3935184 3.93893E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 147 1.46738E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002282 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.17347E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36129E+20 3.0E-06  2.36129E+20 3.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37780E+19 1.3E-07  9.37780E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44477E+20 0.00027  1.33078E+20 0.00021  1.13990E+19 0.00193 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38255E+20 0.00016  2.26856E+20 0.00012  1.13990E+19 0.00193 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38023E+20 0.00036  2.38023E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.69544E+22 0.00026  3.56621E+22 0.00024  1.29230E+21 0.00203 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.50185E+15 0.09080 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38258E+20 0.00016 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16059E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29409E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.22913E-01 0.07890 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.89444E-01 0.00784 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.67177E-04 0.01380 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.20572E+03 0.01527 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99985E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.56283E-01 0.06405 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.56279E-01 0.06405 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51795E+00 2.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99669E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.91876E-01 0.00040  2.01763E-02 0.00039  6.45827E-05 0.00930 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.92273E-01 0.00016 ];
COL_KEFF                  (idx, [1:   2]) = [  9.92132E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.92273E-01 0.00016 ];
ABS_KINF                  (idx, [1:   2]) = [  9.92287E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.43098E+00 0.00017 ];
IMP_ALF                   (idx, [1:   2]) = [  7.43032E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  8.89403E-03 0.00126 ];
IMP_EALF                  (idx, [1:   2]) = [  8.89791E-03 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.15698E-01 0.00164 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.15642E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28819E-03 0.00563  2.66162E-04 0.01924  8.14377E-04 0.01120  5.95694E-04 0.01252  1.29211E-03 0.00891  2.68752E-04 0.02079  5.10987E-05 0.04637 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42583E-01 0.01514  1.22794E-02 0.00478  3.23425E-02 9.6E-05  1.06213E-01 0.00064  2.97473E-01 0.00028  1.20716E+00 0.00579  3.20239E+00 0.04546 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.16143E-03 0.00823  2.58984E-04 0.02733  7.85229E-04 0.01679  5.78878E-04 0.01920  1.23389E-03 0.01312  2.55296E-04 0.03031  4.91551E-05 0.07382 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.39193E-01 0.02390  1.24756E-02 3.7E-05  3.23395E-02 0.00012  1.06157E-01 0.00080  2.97536E-01 0.00039  1.23554E+00 0.00029  6.38422E+00 0.02683 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.13994E-06 0.00114  1.13971E-06 0.00114  1.22464E-06 0.01558 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.13055E-06 0.00106  1.13032E-06 0.00106  1.21447E-06 0.01555 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18765E-03 0.00932  2.63161E-04 0.03071  7.99833E-04 0.01902  5.75208E-04 0.01994  1.23925E-03 0.01460  2.55616E-04 0.03292  5.45779E-05 0.07108 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53929E-01 0.02513  1.24753E-02 5.0E-05  3.23487E-02 0.00019  1.06343E-01 0.00113  2.97405E-01 0.00046  1.23513E+00 0.00039  6.71077E+00 0.03678 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  1.01375E-06 0.01319  1.01351E-06 0.01319  9.38244E-07 0.04817 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  1.00574E-06 0.01319  1.00550E-06 0.01319  9.30383E-07 0.04816 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.70830E-03 0.03973  2.19542E-04 0.13412  7.32817E-04 0.07456  4.41346E-04 0.09165  1.07520E-03 0.06072  1.88683E-04 0.13207  5.07074E-05 0.31390 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52356E-01 0.10162  1.24768E-02 0.00012  3.23559E-02 0.00055  1.06087E-01 0.00308  2.97567E-01 0.00151  1.23544E+00 0.00107  6.26138E+00 0.12253 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.71561E-03 0.03894  2.24297E-04 0.13425  7.44395E-04 0.07406  4.44727E-04 0.09064  1.06875E-03 0.05919  1.86036E-04 0.12845  4.74079E-05 0.31163 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.47663E-01 0.10073  1.24768E-02 0.00012  3.23559E-02 0.00055  1.06048E-01 0.00302  2.97531E-01 0.00151  1.23544E+00 0.00107  6.26138E+00 0.12253 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.69177E+03 0.03812 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.13249E-06 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.12317E-06 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.16297E-03 0.00556 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.79315E+03 0.00554 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  8.02789E-09 0.00121 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.32223E-05 0.00386  7.32177E-05 0.00387  1.36926E-06 0.28609 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  8.20589E-05 0.01676  8.20483E-05 0.01677  1.41227E-06 0.39933 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.89543E-04 0.01368  5.90178E-04 0.01372  4.01152E-04 0.27992 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73348E+01 0.01159 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.87054E+01 0.00021  4.43445E+01 0.00030 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 10:46:00 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03270E+00  1.00483E+00  1.01147E+00  9.92126E-01  9.92309E-01  9.99593E-01  9.88707E-01  9.86832E-01  1.01076E+00  9.86432E-01  1.00700E+00  9.95042E-01  9.94768E-01  9.99627E-01  9.96688E-01  1.00669E+00  1.00645E+00  9.87964E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.26827E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77317E-01 6.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.08598E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.23962E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.78498E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.87585E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.87585E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.01750E+02 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.10468E-02 0.00161  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428661 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42886E+04 0.00054 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42886E+04 0.00054 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.18498E+02 ;
RUNNING_TIME              (idx, 1)        =  1.24978E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.56167E-02  2.38667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.18238E+01  1.11276E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.97667E-02  4.97667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.18250E-01  2.21333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.24898E+01  2.24965E+01 ];
CPU_USAGE                 (idx, 1)        = 9.48147 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97973E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.48730E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 171 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.14878E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.63122E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.85449E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.85795E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31456E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.25851E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46748E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.94415E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.77299E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.88289E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  5.99311E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06125E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17367E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44369E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35104E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.16372E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54278E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.43992E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.37997E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.62307E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.53534E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41415E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.06640E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.68801E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26596E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.12904E-04 -2.42954E+25  2.15210E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.10939E+00 0.00064 ];
TH232_FISS                (idx, [1:   4]) = [  1.66582E+18 0.00388  1.77529E-02 0.00380 ];
U233_FISS                 (idx, [1:   4]) = [  9.21190E+19 0.00047  9.81850E-01 6.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  9.15350E+14 0.16222  9.76170E-06 0.16221 ];
TH232_CAPT                (idx, [1:   4]) = [  1.17412E+20 0.00048  7.95430E-01 0.00023 ];
U233_CAPT                 (idx, [1:   4]) = [  1.38051E+19 0.00135  9.35255E-02 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  4.32693E+14 0.24556  2.94069E-06 0.24544 ];
SM149_CAPT                (idx, [1:   4]) = [  1.92215E+16 0.03556  1.30232E-04 0.03556 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002043 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.20319E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002043 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6114937 6.12112E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3886955 3.89076E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 151 1.50992E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002043 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.58794E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36122E+20 3.0E-06  2.36122E+20 3.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37775E+19 1.3E-07  9.37775E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.47682E+20 0.00028  1.36077E+20 0.00022  1.16058E+19 0.00200 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.41460E+20 0.00017  2.29854E+20 0.00013  1.16058E+19 0.00200 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41144E+20 0.00034  2.41144E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.74468E+22 0.00025  3.61317E+22 0.00023  1.31508E+21 0.00201 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.64333E+15 0.08837 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41464E+20 0.00017 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17711E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29503E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29503E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.54385E-01 0.07328 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.86918E-01 0.00867 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.72718E-04 0.01400 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.08179E+03 0.01664 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99985E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.68622E-01 0.06153 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.68617E-01 0.06153 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51789E+00 2.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99670E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.79473E-01 0.00040  1.99284E-02 0.00039  6.42794E-05 0.00889 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.79076E-01 0.00017 ];
COL_KEFF                  (idx, [1:   2]) = [  9.79250E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.79076E-01 0.00017 ];
ABS_KINF                  (idx, [1:   2]) = [  9.79091E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.42767E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.43001E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  8.92386E-03 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  8.90042E-03 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.16790E-01 0.00169 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.16534E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.32713E-03 0.00537  2.64428E-04 0.01925  8.09156E-04 0.01143  6.20378E-04 0.01257  1.29703E-03 0.00860  2.84488E-04 0.01934  5.16447E-05 0.04502 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.41680E-01 0.01482  1.21374E-02 0.00632  3.23445E-02 0.00010  1.06147E-01 0.00063  2.97644E-01 0.00027  1.20727E+00 0.00579  3.02285E+00 0.04616 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.17716E-03 0.00796  2.50475E-04 0.02843  7.69565E-04 0.01702  5.89748E-04 0.01923  1.24514E-03 0.01286  2.70938E-04 0.02834  5.12953E-05 0.06337 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.48504E-01 0.02267  1.24760E-02 3.6E-05  3.23413E-02 0.00013  1.06136E-01 0.00083  2.97609E-01 0.00039  1.23579E+00 0.00029  5.94550E+00 0.02859 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.14950E-06 0.00108  1.14916E-06 0.00108  1.25465E-06 0.01626 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.12580E-06 0.00103  1.12546E-06 0.00103  1.22907E-06 0.01629 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20574E-03 0.00906  2.54147E-04 0.03305  7.73561E-04 0.01832  6.03676E-04 0.02147  1.24632E-03 0.01438  2.76566E-04 0.02954  5.14669E-05 0.07344 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.45352E-01 0.02460  1.24761E-02 4.6E-05  3.23458E-02 0.00018  1.06178E-01 0.00105  2.97640E-01 0.00049  1.23497E+00 0.00038  6.01053E+00 0.04226 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  1.02502E-06 0.01298  1.02492E-06 0.01298  9.57139E-07 0.05004 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  1.00440E-06 0.01297  1.00429E-06 0.01297  9.37471E-07 0.04987 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06918E-03 0.03799  2.67229E-04 0.11402  8.47972E-04 0.07430  5.44265E-04 0.08734  1.13711E-03 0.06045  2.35911E-04 0.12754  3.66901E-05 0.28797 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.36133E-01 0.09759  1.24751E-02 0.00014  3.23700E-02 0.00055  1.06137E-01 0.00306  2.97292E-01 0.00142  1.23322E+00 0.00118  7.66888E+00 0.10279 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.01156E-03 0.03715  2.66904E-04 0.11189  8.15332E-04 0.07238  5.27070E-04 0.08523  1.12468E-03 0.06015  2.40383E-04 0.12718  3.71900E-05 0.28556 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.35556E-01 0.09727  1.24751E-02 0.00014  3.23711E-02 0.00055  1.06138E-01 0.00305  2.97293E-01 0.00142  1.23322E+00 0.00118  7.66888E+00 0.10279 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.00751E+03 0.03591 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.14302E-06 0.00061 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.11943E-06 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24013E-03 0.00541 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.83531E+03 0.00543 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.92945E-09 0.00102 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.25004E-05 0.00429  7.25090E-05 0.00430  1.25837E-06 0.27805 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.91482E-05 0.01608  7.91787E-05 0.01608  1.06582E-06 0.40737 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.91503E-04 0.01388  4.91845E-04 0.01386  4.10424E-04 0.27760 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70255E+01 0.01154 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.87585E+01 0.00022  4.43517E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 10:58:28 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02701E+00  1.01331E+00  9.98358E-01  9.83025E-01  9.86067E-01  1.00441E+00  9.96014E-01  9.99947E-01  1.01009E+00  9.91212E-01  9.94779E-01  9.98095E-01  1.00820E+00  1.00921E+00  9.89748E-01  9.90149E-01  9.98335E-01  1.00204E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.25785E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77422E-01 6.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.09246E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.24526E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77517E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.87056E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.87056E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.01378E+02 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.01992E-02 0.00161  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428776 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42883E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42883E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.42130E+02 ;
RUNNING_TIME              (idx, 1)        =  2.49684E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.96500E-02  2.40333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.42100E+01  1.23863E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.01750E-01  5.19833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.39767E-01  2.14167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.49605E+01  1.59404E+02 ];
CPU_USAGE                 (idx, 1)        = 9.69744 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97927E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.72416E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.93967E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.71505E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.58584E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.33819E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.64397E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.56878E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51801E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.65281E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.63778E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.71172E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.00029E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.94087E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.63749E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.55110E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.41679E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.37601E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.61133E+18 ;
CS134_ACTIVITY            (idx, 1)        =  6.93563E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.21895E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.42183E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.62605E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.44543E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.59238E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.70011E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61917E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.54542E-04 -1.19329E+26  2.15305E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.11370E+00 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.67181E+18 0.00386  1.78206E-02 0.00379 ];
U233_FISS                 (idx, [1:   4]) = [  9.19698E+19 0.00050  9.80445E-01 7.1E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.25109E+16 0.03313  2.39906E-04 0.03314 ];
TH232_CAPT                (idx, [1:   4]) = [  1.17289E+20 0.00050  7.85272E-01 0.00023 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37773E+19 0.00129  9.22447E-02 0.00124 ];
U235_CAPT                 (idx, [1:   4]) = [  7.99218E+15 0.05444  5.34750E-05 0.05440 ];
SM149_CAPT                (idx, [1:   4]) = [  8.81671E+16 0.01685  5.90162E-04 0.01683 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001837 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.22459E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001837 1.00122E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6143292 6.14974E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3858363 3.86233E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 182 1.82598E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001837 1.00122E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.49949E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36128E+20 3.0E-06  2.36128E+20 3.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37765E+19 1.2E-07  9.37765E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.49451E+20 0.00027  1.37808E+20 0.00021  1.16424E+19 0.00195 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.43227E+20 0.00017  2.31585E+20 0.00013  1.16424E+19 0.00195 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42873E+20 0.00036  2.42873E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.76048E+22 0.00026  3.62852E+22 0.00024  1.31962E+21 0.00198 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.43609E+15 0.07715 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43232E+20 0.00017 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.18427E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29721E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29721E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.47379E-01 0.04946 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07415E-01 0.01016 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.17050E-04 0.01767 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.71269E+03 0.01572 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.71370E-01 0.03904 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.71362E-01 0.03904 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51799E+00 2.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.72510E-01 0.00042  1.97840E-02 0.00040  6.37206E-05 0.00918 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.71982E-01 0.00017 ];
COL_KEFF                  (idx, [1:   2]) = [  9.72316E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.71982E-01 0.00017 ];
ABS_KINF                  (idx, [1:   2]) = [  9.71999E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.40759E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.40911E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.10554E-03 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  9.08870E-03 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.19773E-01 0.00166 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.19103E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.37990E-03 0.00563  2.71074E-04 0.01885  8.22053E-04 0.01172  6.03776E-04 0.01283  1.33428E-03 0.00905  2.92914E-04 0.01949  5.58036E-05 0.04421 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.54635E-01 0.01521  1.23322E-02 0.00407  3.23386E-02 9.4E-05  1.06103E-01 0.00060  2.97538E-01 0.00027  1.20713E+00 0.00579  3.19666E+00 0.04456 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20482E-03 0.00784  2.51798E-04 0.02871  7.83276E-04 0.01640  5.72740E-04 0.01945  1.26658E-03 0.01285  2.81032E-04 0.02800  4.93946E-05 0.06357 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.49077E-01 0.02105  1.24745E-02 4.1E-05  3.23409E-02 0.00014  1.06023E-01 0.00078  2.97653E-01 0.00039  1.23556E+00 0.00030  6.10067E+00 0.02715 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.14048E-06 0.00117  1.14009E-06 0.00117  1.26887E-06 0.01873 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.10900E-06 0.00110  1.10862E-06 0.00110  1.23338E-06 0.01865 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20365E-03 0.00923  2.53694E-04 0.03250  7.65561E-04 0.01858  5.73148E-04 0.02093  1.28624E-03 0.01412  2.71846E-04 0.03154  5.31633E-05 0.07135 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.53119E-01 0.02500  1.24757E-02 4.8E-05  3.23357E-02 0.00017  1.06005E-01 0.00107  2.97481E-01 0.00045  1.23523E+00 0.00043  6.20592E+00 0.03971 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  1.01160E-06 0.01304  1.01121E-06 0.01304  9.87802E-07 0.04996 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.83817E-07 0.01303  9.83440E-07 0.01303  9.59924E-07 0.04990 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91575E-03 0.03999  2.24033E-04 0.13247  7.25962E-04 0.07547  4.81810E-04 0.09236  1.19929E-03 0.06110  2.20612E-04 0.13791  6.40378E-05 0.28853 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.60037E-01 0.08886  1.24761E-02 0.00013  3.23103E-02 0.00037  1.06127E-01 0.00296  2.97611E-01 0.00148  1.23625E+00 0.00161  7.91215E+00 0.08101 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92593E-03 0.03935  2.23461E-04 0.13064  7.31773E-04 0.07396  4.88853E-04 0.09025  1.19183E-03 0.06091  2.28281E-04 0.13394  6.17287E-05 0.28013 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.54748E-01 0.08671  1.24761E-02 0.00013  3.23103E-02 0.00037  1.06137E-01 0.00297  2.97596E-01 0.00147  1.23615E+00 0.00161  7.91215E+00 0.08101 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.91133E+03 0.03851 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.13515E-06 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.10381E-06 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.27206E-03 0.00574 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.88333E+03 0.00578 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.71966E-09 0.00085 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.97561E-05 0.00652  6.98297E-05 0.00652  7.71931E-07 0.32234 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.23399E-05 0.02014  7.22258E-05 0.02022  1.32053E-06 0.39532 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.30531E-04 0.01737  3.30644E-04 0.01746  2.96950E-04 0.31644 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70091E+01 0.01110 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.87056E+01 0.00022  4.42377E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 11:11:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02630E+00  9.89498E-01  1.00541E+00  9.90573E-01  9.99665E-01  9.94873E-01  9.99619E-01  9.97526E-01  1.00857E+00  9.95148E-01  1.00531E+00  9.89270E-01  9.98933E-01  9.97663E-01  1.00075E+00  1.01017E+00  1.00155E+00  9.89167E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.25125E-02 0.00027  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77487E-01 6.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.09953E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.25175E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77330E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.85578E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.85578E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.00772E+02 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.91858E-02 0.00165  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429114 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42899E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42899E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.69131E+02 ;
RUNNING_TIME              (idx, 1)        =  3.77834E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.02650E-01  2.30000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.69337E+01  1.27237E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.54200E-01  5.24500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.60967E-01  2.11167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.77752E+01  1.74854E+02 ];
CPU_USAGE                 (idx, 1)        = 9.76968 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97998E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.80303E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.06413E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74029E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.05920E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.34730E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.64847E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.68395E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.54264E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.71202E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.80999E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.83503E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.00823E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.87575E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.80173E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.50440E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.46757E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.41559E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.66439E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.26230E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.43865E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.50260E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.69223E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.81420E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.72813E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.69392E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32022E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.03714E-03 -2.23177E+26  2.15409E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.10425E+00 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.65074E+18 0.00380  1.76002E-02 0.00375 ];
U233_FISS                 (idx, [1:   4]) = [  9.17934E+19 0.00050  9.78732E-01 7.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.07539E+16 0.01732  8.61042E-04 0.01732 ];
TH232_CAPT                (idx, [1:   4]) = [  1.15707E+20 0.00048  7.79253E-01 0.00022 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37618E+19 0.00124  9.26839E-02 0.00119 ];
U235_CAPT                 (idx, [1:   4]) = [  3.03287E+16 0.02860  2.04325E-04 0.02860 ];
SM149_CAPT                (idx, [1:   4]) = [  1.38446E+17 0.01287  9.32448E-04 0.01287 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002939 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.19724E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002939 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6130353 6.13603E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3872411 3.87577E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 175 1.74997E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002939 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.82311E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36141E+20 3.1E-06  2.36141E+20 3.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 1.2E-07  9.37751E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.48419E+20 0.00026  1.36933E+20 0.00019  1.14856E+19 0.00198 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.42194E+20 0.00016  2.30708E+20 0.00012  1.14856E+19 0.00198 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41988E+20 0.00035  2.41988E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.73057E+22 0.00024  3.60016E+22 0.00022  1.30409E+21 0.00198 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.23538E+15 0.08939 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.42198E+20 0.00016 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17635E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29687E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29687E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.38357E-01 0.04744 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.32332E-01 0.00960 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.61066E-04 0.01897 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.73198E+03 0.01561 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 1.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.80995E-01 0.03837 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.80986E-01 0.03837 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51816E+00 3.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.76010E-01 0.00040  1.98546E-02 0.00039  6.31954E-05 0.00943 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.76176E-01 0.00016 ];
COL_KEFF                  (idx, [1:   2]) = [  9.75916E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.76176E-01 0.00016 ];
ABS_KINF                  (idx, [1:   2]) = [  9.76193E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.38383E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.38364E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.32404E-03 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  9.32300E-03 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.21582E-01 0.00165 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.21668E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33099E-03 0.00568  2.74150E-04 0.01924  8.17689E-04 0.01125  5.96179E-04 0.01249  1.31450E-03 0.00861  2.72576E-04 0.01949  5.59006E-05 0.04322 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50456E-01 0.01521  1.21728E-02 0.00597  3.23402E-02 0.00010  1.05975E-01 0.00155  2.97378E-01 0.00026  1.20903E+00 0.00560  3.33295E+00 0.04370 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.16846E-03 0.00835  2.61011E-04 0.02776  7.71093E-04 0.01632  5.76281E-04 0.01917  1.25038E-03 0.01268  2.58297E-04 0.02898  5.13962E-05 0.05956 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46182E-01 0.02183  1.24755E-02 3.6E-05  3.23376E-02 0.00013  1.06105E-01 0.00076  2.97309E-01 0.00036  1.23560E+00 0.00032  6.22346E+00 0.02660 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.11519E-06 0.00113  1.11489E-06 0.00114  1.21060E-06 0.01923 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.08831E-06 0.00105  1.08801E-06 0.00106  1.18181E-06 0.01928 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18563E-03 0.00962  2.64702E-04 0.03128  7.94413E-04 0.01877  5.66851E-04 0.02085  1.24779E-03 0.01456  2.64137E-04 0.03305  4.77321E-05 0.07127 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.43158E-01 0.02476  1.24759E-02 4.8E-05  3.23430E-02 0.00019  1.06080E-01 0.00105  2.97277E-01 0.00046  1.23606E+00 0.00041  6.60774E+00 0.03893 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.85109E-07 0.01301  9.84830E-07 0.01300  9.26126E-07 0.04939 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.61409E-07 0.01299  9.61136E-07 0.01299  9.04103E-07 0.04946 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93687E-03 0.04079  2.88679E-04 0.11566  6.91475E-04 0.08225  5.15143E-04 0.09459  1.16080E-03 0.06497  2.49276E-04 0.13286  3.15069E-05 0.27737 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.60109E-01 0.09652  1.24766E-02 0.00011  3.23318E-02 0.00047  1.06776E-01 0.00354  2.97056E-01 0.00137  1.23610E+00 0.00097  6.83834E+00 0.10638 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.95722E-03 0.03988  2.90153E-04 0.11254  6.98470E-04 0.08033  5.07093E-04 0.09209  1.17725E-03 0.06370  2.51610E-04 0.13158  3.26497E-05 0.26752 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.63028E-01 0.09720  1.24766E-02 0.00011  3.23319E-02 0.00047  1.06784E-01 0.00355  2.97127E-01 0.00138  1.23610E+00 0.00097  6.83834E+00 0.10638 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.99842E+03 0.03899 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.10744E-06 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.08075E-06 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26179E-03 0.00542 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.94547E+03 0.00539 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.59486E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.90505E-05 0.00747  6.90295E-05 0.00749  5.63911E-07 0.41718 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.66006E-05 0.02373  6.66048E-05 0.02373  6.61535E-07 0.45975 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.70670E-04 0.01862  2.70975E-04 0.01862  1.92919E-04 0.40798 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71331E+01 0.01155 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.85578E+01 0.00022  4.40523E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 11:24:41 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04036E+00  9.98778E-01  9.87906E-01  9.93245E-01  1.01338E+00  9.95543E-01  9.90329E-01  1.01181E+00  9.94662E-01  1.01161E+00  9.87460E-01  9.89518E-01  9.88534E-01  1.00596E+00  9.85356E-01  9.95428E-01  1.00575E+00  1.00438E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.24620E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77538E-01 6.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.11035E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.26198E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76269E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.83655E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.83654E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.99060E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.88864E-02 0.00159  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428827 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42878E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42878E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.01966E+02 ;
RUNNING_TIME              (idx, 1)        =  5.11837E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.30250E-01  2.76000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.02405E+01  1.33068E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.04833E-01  5.06333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.85417E-01  2.43667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.11756E+01  1.79195E+02 ];
CPU_USAGE                 (idx, 1)        = 9.80715 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97962E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84322E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.10656E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75501E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.26760E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.30911E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.61835E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.76473E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56053E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09188E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.95200E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.04684E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.92355E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.86519E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.95951E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.98660E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.50444E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.44498E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.70273E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.14058E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.85790E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.51420E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.74629E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.79691E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.72061E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.68139E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64043E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.93863E-03 -4.17164E+26  2.15603E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.08759E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.61298E+18 0.00387  1.71974E-02 0.00380 ];
U233_FISS                 (idx, [1:   4]) = [  9.13987E+19 0.00050  9.74578E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.90560E+17 0.00918  3.09798E-03 0.00916 ];
TH232_CAPT                (idx, [1:   4]) = [  1.12844E+20 0.00050  7.69239E-01 0.00024 ];
U233_CAPT                 (idx, [1:   4]) = [  1.36306E+19 0.00132  9.29207E-02 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  1.05184E+17 0.01573  7.16707E-04 0.01566 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84455E+17 0.01143  1.25766E-03 0.01143 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001434 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.18643E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001434 1.00119E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6100656 6.10726E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3900611 3.90444E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 167 1.67999E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001434 1.00119E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.80677E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36155E+20 3.2E-06  2.36155E+20 3.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37712E+19 1.2E-07  9.37712E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46597E+20 0.00026  1.35217E+20 0.00020  1.13802E+19 0.00199 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40369E+20 0.00016  2.28989E+20 0.00012  1.13802E+19 0.00199 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40199E+20 0.00035  2.40199E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68077E+22 0.00025  3.55096E+22 0.00023  1.29805E+21 0.00206 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.03855E+15 0.08930 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.40373E+20 0.00016 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16302E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29268E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29268E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.40338E-01 0.04681 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17506E-01 0.01227 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.23024E-04 0.02231 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.76402E+03 0.01568 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99983E-01 1.5E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.81776E-01 0.03859 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.81767E-01 0.03859 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51842E+00 3.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99683E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.83413E-01 0.00041  2.00039E-02 0.00040  6.33806E-05 0.00858 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.83645E-01 0.00016 ];
COL_KEFF                  (idx, [1:   2]) = [  9.83248E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.83645E-01 0.00016 ];
ABS_KINF                  (idx, [1:   2]) = [  9.83662E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.34153E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.34095E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.72691E-03 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  9.72975E-03 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.25705E-01 0.00167 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.26083E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30977E-03 0.00551  2.72666E-04 0.01891  7.98437E-04 0.01155  6.01989E-04 0.01293  1.30229E-03 0.00919  2.79298E-04 0.01878  5.50833E-05 0.04141 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.59196E-01 0.01460  1.22081E-02 0.00560  3.23311E-02 0.00011  1.06191E-01 0.00060  2.97645E-01 0.00029  1.22323E+00 0.00409  3.59163E+00 0.04214 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.15572E-03 0.00825  2.56881E-04 0.02685  7.55444E-04 0.01699  5.79615E-04 0.01970  1.25226E-03 0.01343  2.59360E-04 0.02811  5.21584E-05 0.06459 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.50561E-01 0.02141  1.24754E-02 3.7E-05  3.23263E-02 0.00014  1.06151E-01 0.00076  2.97767E-01 0.00041  1.23733E+00 0.00045  6.54481E+00 0.02507 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.07388E-06 0.00113  1.07366E-06 0.00113  1.13969E-06 0.01768 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.05595E-06 0.00107  1.05574E-06 0.00107  1.12054E-06 0.01765 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.15133E-03 0.00883  2.61436E-04 0.03071  7.59054E-04 0.01910  5.71174E-04 0.02243  1.24516E-03 0.01386  2.69657E-04 0.03056  4.48458E-05 0.07568 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.45391E-01 0.02459  1.24759E-02 4.6E-05  3.23338E-02 0.00020  1.06204E-01 0.00119  2.97490E-01 0.00045  1.23681E+00 0.00050  6.53724E+00 0.04101 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.50087E-07 0.01305  9.49997E-07 0.01305  8.27313E-07 0.05276 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.34493E-07 0.01305  9.34405E-07 0.01305  8.13792E-07 0.05277 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.81138E-03 0.03930  2.64247E-04 0.12601  6.93760E-04 0.07807  5.56078E-04 0.08707  1.01149E-03 0.06126  2.69652E-04 0.11962  1.61565E-05 0.35380 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.10765E-01 0.08130  1.24764E-02 0.00012  3.23013E-02 0.00045  1.05916E-01 0.00274  2.97215E-01 0.00140  1.23783E+00 0.00135  7.25185E+00 0.13122 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81974E-03 0.03892  2.52169E-04 0.12290  6.92930E-04 0.07688  5.60157E-04 0.08605  1.01762E-03 0.06036  2.78900E-04 0.11788  1.79656E-05 0.32632 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.13171E-01 0.07907  1.24764E-02 0.00012  3.23030E-02 0.00045  1.05901E-01 0.00272  2.97204E-01 0.00140  1.23782E+00 0.00135  7.25185E+00 0.13122 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.98940E+03 0.03745 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.06695E-06 0.00061 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.04913E-06 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.15963E-03 0.00607 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.96258E+03 0.00614 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.48476E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.56964E-05 0.01062  6.56944E-05 0.01062  4.58055E-07 0.44796 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.43894E-05 0.02702  6.43407E-05 0.02706  3.12262E-07 0.76363 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.32205E-04 0.02190  2.32469E-04 0.02191  1.52309E-04 0.44985 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72416E+01 0.01205 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.83654E+01 0.00021  4.36793E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 11:38:21 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02529E+00  1.00935E+00  9.93294E-01  9.90755E-01  9.97308E-01  1.00693E+00  1.00108E+00  1.00188E+00  9.97068E-01  9.99264E-01  9.87748E-01  9.85987E-01  1.00298E+00  9.91053E-01  9.94747E-01  1.01004E+00  9.97891E-01  1.00734E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.24594E-02 0.00027  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77541E-01 6.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.11447E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.26601E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75982E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.81432E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.81432E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.92645E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.78352E-02 0.00166  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429051 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42901E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42901E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.37453E+02 ;
RUNNING_TIME              (idx, 1)        =  6.48554E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.59017E-01  2.87667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.38136E+01  1.35731E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.55533E-01  5.07000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.11367E-01  2.58667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.48468E+01  1.85324E+02 ];
CPU_USAGE                 (idx, 1)        = 9.82883 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98029E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86640E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.05380E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74632E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.70514E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.25743E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.57898E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.76393E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55607E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.17782E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.00871E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.35052E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.71663E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.41026E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.03670E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.42830E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.48116E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.42791E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.67831E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.76904E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.24026E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.46105E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.73297E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.92078E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.60321E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.66851E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96065E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.87594E-03 -6.18860E+26  2.15804E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.07265E+00 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.59631E+18 0.00378  1.70134E-02 0.00374 ];
U233_FISS                 (idx, [1:   4]) = [  9.09919E+19 0.00047  9.69809E-01 8.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  5.56705E+17 0.00643  5.93337E-03 0.00640 ];
TH232_CAPT                (idx, [1:   4]) = [  1.10423E+20 0.00050  7.62556E-01 0.00024 ];
U233_CAPT                 (idx, [1:   4]) = [  1.35399E+19 0.00138  9.35033E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  2.03852E+17 0.01071  1.40771E-03 0.01069 ];
SM149_CAPT                (idx, [1:   4]) = [  1.97033E+17 0.01112  1.36082E-03 0.01114 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003104 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16778E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003104 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6069933 6.07516E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3933009 3.93636E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 162 1.61872E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003104 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.30385E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36157E+20 3.2E-06  2.36157E+20 3.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37666E+19 1.1E-07  9.37666E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44753E+20 0.00025  1.33600E+20 0.00019  1.11534E+19 0.00199 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38520E+20 0.00015  2.27366E+20 0.00011  1.11534E+19 0.00199 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38358E+20 0.00034  2.38358E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.63503E+22 0.00024  3.50780E+22 0.00023  1.27238E+21 0.00199 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.86093E+15 0.08403 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38523E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14877E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28796E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28796E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.49375E-01 0.04650 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30188E-01 0.01181 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.07186E-04 0.02154 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.78684E+03 0.01569 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.71638E-01 0.03971 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.71631E-01 0.03971 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51856E+00 3.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99693E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.91459E-01 0.00039  2.01669E-02 0.00039  6.54105E-05 0.00887 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.91269E-01 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  9.90844E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.91269E-01 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  9.91285E-01 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.31051E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.30802E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.00333E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.00551E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.29294E-01 0.00167 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.29592E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30529E-03 0.00547  2.70738E-04 0.02030  8.00196E-04 0.01170  6.00012E-04 0.01284  1.30278E-03 0.00872  2.76725E-04 0.01879  5.48375E-05 0.04280 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.58767E-01 0.01496  1.21193E-02 0.00649  3.23220E-02 9.9E-05  1.06213E-01 0.00062  2.97893E-01 0.00028  1.21686E+00 0.00501  3.59033E+00 0.04198 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.18890E-03 0.00778  2.60119E-04 0.02944  7.76500E-04 0.01664  5.75601E-04 0.01879  1.25412E-03 0.01218  2.69942E-04 0.02884  5.26120E-05 0.06490 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.61884E-01 0.02374  1.24760E-02 3.3E-05  3.23176E-02 0.00013  1.06208E-01 0.00083  2.97834E-01 0.00037  1.23825E+00 0.00044  6.54413E+00 0.02489 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.04319E-06 0.00108  1.04291E-06 0.00108  1.13274E-06 0.01604 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.03420E-06 0.00105  1.03392E-06 0.00106  1.12313E-06 0.01604 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23013E-03 0.00900  2.74544E-04 0.03142  7.83705E-04 0.01858  5.71345E-04 0.02145  1.27501E-03 0.01429  2.74776E-04 0.03211  5.07429E-05 0.07371 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.58203E-01 0.02617  1.24753E-02 5.1E-05  3.23244E-02 0.00018  1.06188E-01 0.00107  2.98033E-01 0.00050  1.23920E+00 0.00060  6.70432E+00 0.03661 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.22697E-07 0.01305  9.22496E-07 0.01305  8.70902E-07 0.04717 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.14929E-07 0.01304  9.14727E-07 0.01304  8.64016E-07 0.04722 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.70819E-03 0.03856  2.19555E-04 0.12970  6.63322E-04 0.07627  4.83020E-04 0.09005  1.03954E-03 0.06335  2.57693E-04 0.11010  4.50576E-05 0.35394 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.73580E-01 0.09872  1.24758E-02 0.00013  3.22978E-02 0.00043  1.06721E-01 0.00346  2.98061E-01 0.00151  1.23915E+00 0.00152  7.24715E+00 0.12224 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.73375E-03 0.03794  2.17397E-04 0.12784  6.75870E-04 0.07455  4.91323E-04 0.08966  1.04857E-03 0.06148  2.59730E-04 0.10887  4.08578E-05 0.34566 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.68397E-01 0.09580  1.24757E-02 0.00013  3.22980E-02 0.00043  1.06727E-01 0.00348  2.98020E-01 0.00149  1.23915E+00 0.00152  7.20232E+00 0.12319 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.96372E+03 0.03659 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.03693E-06 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.02796E-06 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20856E-03 0.00539 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.09503E+03 0.00541 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.40917E-09 0.00070 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.57182E-05 0.01065  6.57311E-05 0.01066  8.30218E-07 0.35539 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.20508E-05 0.02593  6.19711E-05 0.02596  1.06774E-06 0.41084 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.15278E-04 0.02116  2.15086E-04 0.02121  2.89038E-04 0.33407 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70779E+01 0.01208 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.81432E+01 0.00021  4.33749E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 11:52:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02664E+00  1.00302E+00  9.92400E-01  9.84123E-01  9.88410E-01  9.95429E-01  1.01046E+00  1.01021E+00  9.93314E-01  1.00690E+00  9.94377E-01  9.96515E-01  9.96595E-01  1.00329E+00  1.00334E+00  9.92743E-01  1.00165E+00  1.00058E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.24476E-02 0.00027  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77552E-01 6.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.12208E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.27337E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75624E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.78620E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.78619E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.83557E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.74996E-02 0.00166  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429005 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42881E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42881E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.73918E+02 ;
RUNNING_TIME              (idx, 1)        =  7.86253E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.83567E-01  2.45500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.74841E+01  1.36705E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.07550E-01  5.20167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.33367E-01  2.19167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.86173E+01  1.87815E+02 ];
CPU_USAGE                 (idx, 1)        = 9.84312 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98173E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88175E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.06261E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75780E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.35073E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.20484E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.53750E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82558E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57198E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.35346E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.14849E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  6.30508E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.52963E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  7.17814E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.19452E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.21911E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.51099E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.45070E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.70854E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.57429E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.92161E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.45373E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.80663E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.99059E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.53949E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.65024E+16 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60108E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -4.63051E-03 -9.96419E+26  2.16182E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.05203E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.56454E+18 0.00385  1.66773E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  8.99917E+19 0.00052  9.59404E-01 9.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.20458E+18 0.00429  1.28439E-02 0.00431 ];
PU239_FISS                (idx, [1:   4]) = [  9.46632E+13 0.49895  1.00203E-06 0.49897 ];
PU240_FISS                (idx, [1:   4]) = [  2.36057E+13 1.00000  2.49750E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.06767E+20 0.00053  7.50700E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33778E+19 0.00132  9.40642E-02 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  4.47244E+17 0.00725  3.14521E-03 0.00727 ];
PU239_CAPT                (idx, [1:   4]) = [  2.37035E+13 1.00000  1.68543E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  2.00417E+17 0.01084  1.40909E-03 0.01083 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001685 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.18111E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001685 1.00118E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6026610 6.03281E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3974918 3.97884E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 157 1.57377E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001685 1.00118E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.45058E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36143E+20 3.4E-06  2.36143E+20 3.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37565E+19 1.1E-07  9.37565E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42173E+20 0.00026  1.31204E+20 0.00018  1.09685E+19 0.00205 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35929E+20 0.00016  2.24961E+20 0.00011  1.09685E+19 0.00205 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.35749E+20 0.00036  2.35749E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.57268E+22 0.00024  3.44666E+22 0.00022  1.26017E+21 0.00210 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.71244E+15 0.08868 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.35933E+20 0.00016 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12957E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27754E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27754E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.85170E-01 0.05026 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19854E-01 0.01184 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.11168E-04 0.02075 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.99735E+03 0.01518 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.46657E-01 0.04218 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.46650E-01 0.04218 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51869E+00 3.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99715E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.00202E+00 0.00043  2.03859E-02 0.00041  6.59499E-05 0.00872 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.00209E+00 0.00016 ];
COL_KEFF                  (idx, [1:   2]) = [  1.00176E+00 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.00209E+00 0.00016 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00211E+00 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.25965E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.25939E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.05569E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.05564E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.35839E-01 0.00163 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.35760E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.31585E-03 0.00547  2.65301E-04 0.01901  8.02225E-04 0.01164  6.06818E-04 0.01315  1.31011E-03 0.00855  2.79178E-04 0.01919  5.22145E-05 0.04413 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.55061E-01 0.01546  1.22618E-02 0.00500  3.23138E-02 0.00011  1.06285E-01 0.00063  2.98125E-01 0.00027  1.21431E+00 0.00562  3.39639E+00 0.04392 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20513E-03 0.00797  2.52934E-04 0.02831  7.74571E-04 0.01581  5.87737E-04 0.01851  1.26640E-03 0.01296  2.69409E-04 0.02785  5.40727E-05 0.06211 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.60157E-01 0.02106  1.24755E-02 3.7E-05  3.23156E-02 0.00014  1.06275E-01 0.00079  2.98178E-01 0.00038  1.24045E+00 0.00052  6.62691E+00 0.02458 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.00109E-06 0.00116  1.00079E-06 0.00116  1.08956E-06 0.01721 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.00298E-06 0.00107  1.00268E-06 0.00107  1.09190E-06 0.01724 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22461E-03 0.00885  2.53859E-04 0.03316  7.80297E-04 0.01844  5.91156E-04 0.02160  1.27925E-03 0.01427  2.67987E-04 0.03189  5.20598E-05 0.07051 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.55911E-01 0.02516  1.24754E-02 5.1E-05  3.23254E-02 0.00022  1.06318E-01 0.00111  2.98075E-01 0.00048  1.24059E+00 0.00070  6.49198E+00 0.03723 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.86259E-07 0.01304  8.85988E-07 0.01304  8.24722E-07 0.04665 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.88177E-07 0.01303  8.87908E-07 0.01303  8.25822E-07 0.04659 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.79475E-03 0.03804  2.55488E-04 0.12832  6.72145E-04 0.07556  5.55218E-04 0.08503  1.02383E-03 0.05883  2.53195E-04 0.10995  3.48751E-05 0.26818 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.39770E-01 0.07677  1.24733E-02 0.00016  3.23224E-02 0.00056  1.06246E-01 0.00286  2.98345E-01 0.00159  1.24644E+00 0.00234  6.09258E+00 0.11836 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.78992E-03 0.03769  2.49984E-04 0.12585  6.71467E-04 0.07608  5.55148E-04 0.08279  1.02594E-03 0.05812  2.49601E-04 0.10991  3.77753E-05 0.26808 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.37325E-01 0.07657  1.24734E-02 0.00016  3.23218E-02 0.00056  1.06245E-01 0.00285  2.98335E-01 0.00158  1.24635E+00 0.00232  6.09258E+00 0.11836 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.19487E+03 0.03670 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.95627E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.97514E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24040E-03 0.00542 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.25444E+03 0.00537 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.34301E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70745E-05 0.00996  6.71055E-05 0.00997  5.20822E-07 0.40741 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.37894E-05 0.02793  6.36137E-05 0.02795  7.48339E-07 0.75108 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.19869E-04 0.02055  2.19995E-04 0.02064  2.01383E-04 0.41997 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72857E+01 0.01142 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.78619E+01 0.00023  4.28711E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 12:05:38 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02713E+00  1.00085E+00  9.92050E-01  9.92519E-01  9.95412E-01  9.98659E-01  1.00300E+00  9.98784E-01  9.96944E-01  9.95686E-01  9.93525E-01  1.00107E+00  9.99699E-01  1.00531E+00  9.95332E-01  1.00473E+00  1.00002E+00  9.99276E-01  ];
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

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 8.1E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.24903E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77510E-01 6.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.13108E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.28245E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75189E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.74568E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.74568E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.71219E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.71321E-02 0.00163  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428808 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42890E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42890E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  9.07837E+02 ;
RUNNING_TIME              (idx, 1)        =  9.21456E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.10183E-01  2.66167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  9.09026E+01  1.34184E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.57650E-01  5.01000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.55650E-01  2.21667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  9.21377E+01  1.88368E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85220 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98147E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89195E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.05665E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.77069E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.12143E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.12147E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.47123E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.90323E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.59177E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.86082E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.42853E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.87368E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.29934E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.57199E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.49539E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.36541E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.53066E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.46323E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.72638E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.36738E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.31336E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.42737E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.93792E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.89401E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.41476E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.62908E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32022E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -8.65946E-03 -1.86339E+27  2.17049E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02601E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.49660E+18 0.00398  1.59566E-02 0.00396 ];
U233_FISS                 (idx, [1:   4]) = [  8.76176E+19 0.00050  9.34140E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.91721E+18 0.00292  3.11009E-02 0.00286 ];
PU239_FISS                (idx, [1:   4]) = [  2.29916E+15 0.10421  2.45217E-05 0.10430 ];
PU240_FISS                (idx, [1:   4]) = [  2.33937E+13 1.00000  2.44744E-07 1.00000 ];
PU241_FISS                (idx, [1:   4]) = [  4.64976E+13 0.70660  4.92527E-07 0.70663 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01447E+20 0.00049  7.28797E-01 0.00025 ];
U233_CAPT                 (idx, [1:   4]) = [  1.30100E+19 0.00131  9.34663E-02 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  1.05117E+18 0.00455  7.55228E-03 0.00455 ];
U238_CAPT                 (idx, [1:   4]) = [  4.63830E+13 0.70660  3.32712E-07 0.70660 ];
PU239_CAPT                (idx, [1:   4]) = [  1.25575E+15 0.13584  9.01927E-06 0.13591 ];
PU240_CAPT                (idx, [1:   4]) = [  1.15059E+14 0.44594  8.28905E-07 0.44594 ];
PU241_CAPT                (idx, [1:   4]) = [  2.33566E+13 1.00000  1.69122E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.97875E+17 0.01148  1.42136E-03 0.01146 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002322 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16834E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002322 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5975399 5.98122E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4026762 4.03031E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 161 1.61069E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002322 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.63913E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36064E+20 3.5E-06  2.36064E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37309E+19 1.0E-07  9.37309E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.39217E+20 0.00025  1.28441E+20 0.00019  1.07760E+19 0.00204 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.32948E+20 0.00015  2.22172E+20 0.00011  1.07760E+19 0.00204 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.32726E+20 0.00032  2.32726E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49876E+22 0.00023  3.37402E+22 0.00021  1.24741E+21 0.00209 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.74817E+15 0.08557 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.32952E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10564E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.24979E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.24979E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.71327E-01 0.04768 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.30505E-01 0.01138 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.05124E-04 0.02126 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.05373E+03 0.01421 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.87183E-01 0.03937 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.87175E-01 0.03937 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51853E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99769E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.01484E+00 0.00041  2.06485E-02 0.00039  6.66626E-05 0.00890 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.01455E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.01442E+00 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.01455E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01456E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.18747E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.18603E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.13470E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.13594E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.46460E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46623E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.29847E-03 0.00540  2.57543E-04 0.01895  7.91306E-04 0.01091  5.94950E-04 0.01278  1.31067E-03 0.00836  2.87084E-04 0.01859  5.69207E-05 0.04113 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.74018E-01 0.01496  1.22086E-02 0.00560  3.22847E-02 0.00011  1.06399E-01 0.00060  2.98942E-01 0.00028  1.23486E+00 0.00386  3.94155E+00 0.03949 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24288E-03 0.00751  2.53708E-04 0.02639  7.82841E-04 0.01550  5.72871E-04 0.01831  1.27959E-03 0.01254  2.90135E-04 0.02727  6.37342E-05 0.06222 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.87005E-01 0.02276  1.24757E-02 3.9E-05  3.22842E-02 0.00016  1.06283E-01 0.00075  2.98941E-01 0.00039  1.24718E+00 0.00076  6.96796E+00 0.02190 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.47254E-07 0.00119  9.46938E-07 0.00119  1.04432E-06 0.01664 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.61186E-07 0.00111  9.60867E-07 0.00111  1.05929E-06 0.01658 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22584E-03 0.00913  2.50293E-04 0.03178  7.76422E-04 0.01772  5.64351E-04 0.02090  1.30043E-03 0.01397  2.77894E-04 0.03041  5.64408E-05 0.06635 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.76369E-01 0.02511  1.24754E-02 5.5E-05  3.22845E-02 0.00021  1.06280E-01 0.00101  2.99010E-01 0.00049  1.24817E+00 0.00103  6.91436E+00 0.03209 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.36611E-07 0.01298  8.36644E-07 0.01298  7.35260E-07 0.04819 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.49380E-07 0.01298  8.49410E-07 0.01298  7.46802E-07 0.04835 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.77458E-03 0.03780  2.54068E-04 0.12981  7.43269E-04 0.07254  4.80840E-04 0.09017  1.00074E-03 0.05825  2.58551E-04 0.11426  3.71150E-05 0.31756 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.44671E-01 0.08893  1.24739E-02 0.00016  3.22835E-02 0.00056  1.06384E-01 0.00288  2.99177E-01 0.00161  1.24347E+00 0.00268  8.62634E+00 0.07403 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.81764E-03 0.03717  2.47339E-04 0.12959  7.51667E-04 0.07103  4.88184E-04 0.08687  1.01813E-03 0.05768  2.70866E-04 0.11273  4.14509E-05 0.30984 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.49219E-01 0.08960  1.24738E-02 0.00016  3.22855E-02 0.00056  1.06391E-01 0.00288  2.99177E-01 0.00160  1.24362E+00 0.00268  8.62634E+00 0.07403 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.33052E+03 0.03595 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.41238E-07 0.00061 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.55094E-07 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21089E-03 0.00589 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.41189E+03 0.00589 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.23913E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67877E-05 0.00948  6.67682E-05 0.00948  7.99700E-07 0.35908 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.17537E-05 0.02664  6.18400E-05 0.02665  4.90618E-07 0.40888 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.15309E-04 0.02078  2.15221E-04 0.02076  2.50155E-04 0.35281 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65635E+01 0.01084 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.74568E+01 0.00023  4.21480E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 12:19:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04339E+00  9.92597E-01  9.93340E-01  1.00549E+00  9.96531E-01  9.97949E-01  1.00029E+00  9.83688E-01  1.00243E+00  1.00391E+00  1.00011E+00  9.97560E-01  1.00180E+00  1.00172E+00  9.96142E-01  9.84580E-01  1.00404E+00  9.94427E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.25858E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77414E-01 6.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.13221E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.28419E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75542E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.70696E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.70696E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.62533E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.69587E-02 0.00165  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428817 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42899E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42899E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.04150E+03 ;
RUNNING_TIME              (idx, 1)        =  1.05642E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.35317E-01  2.51333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.04294E+02  1.33914E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.10250E-01  5.26000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.77000E-01  2.12667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.05635E+02  1.86611E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85877 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98153E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89944E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.97196E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75932E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.65388E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02442E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39682E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.91584E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.58816E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.43733E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.85268E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.33978E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.23994E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.06652E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.92256E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.42983E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.46675E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.40928E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.65575E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.62627E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.35171E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.33571E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01409E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.12740E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.19797E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61874E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64369E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.74987E-02 -3.76548E+27  2.18951E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01131E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.43903E+18 0.00405  1.53598E-02 0.00399 ];
U233_FISS                 (idx, [1:   4]) = [  8.42071E+19 0.00052  8.98844E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  5.31020E+18 0.00215  5.66785E-02 0.00204 ];
PU239_FISS                (idx, [1:   4]) = [  3.77106E+16 0.02546  4.02723E-04 0.02551 ];
PU240_FISS                (idx, [1:   4]) = [  1.38900E+15 0.12781  1.48019E-05 0.12783 ];
PU241_FISS                (idx, [1:   4]) = [  1.52532E+15 0.12494  1.62587E-05 0.12473 ];
TH232_CAPT                (idx, [1:   4]) = [  9.68470E+19 0.00052  7.02702E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.24939E+19 0.00134  9.06563E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  1.91536E+18 0.00342  1.38979E-02 0.00341 ];
U238_CAPT                 (idx, [1:   4]) = [  1.83976E+14 0.35184  1.33519E-06 0.35182 ];
PU239_CAPT                (idx, [1:   4]) = [  1.85777E+16 0.03564  1.34832E-04 0.03562 ];
PU240_CAPT                (idx, [1:   4]) = [  3.14879E+15 0.08882  2.28399E-05 0.08876 ];
PU241_CAPT                (idx, [1:   4]) = [  3.74621E+14 0.26320  2.70622E-06 0.26294 ];
SM149_CAPT                (idx, [1:   4]) = [  1.95030E+17 0.01079  1.41530E-03 0.01080 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002954 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13343E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002954 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5954764 5.95988E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4048011 4.05128E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 179 1.78808E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002954 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -7.26432E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35956E+20 3.7E-06  2.35956E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36907E+19 1.0E-07  9.36907E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37893E+20 0.00026  1.27286E+20 0.00019  1.06068E+19 0.00211 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31583E+20 0.00015  2.20977E+20 0.00011  1.06068E+19 0.00211 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.31248E+20 0.00035  2.31248E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45626E+22 0.00024  3.33317E+22 0.00022  1.23090E+21 0.00211 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.13834E+15 0.08200 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.31588E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08961E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.20336E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.20336E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.46123E-01 0.05155 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24531E-01 0.01115 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.23784E-04 0.02079 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.03962E+03 0.01597 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 1.5E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.50117E-01 0.04255 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.50108E-01 0.04255 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51846E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99855E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02018E+00 0.00041  2.07520E-02 0.00040  6.97772E-05 0.00854 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02003E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02045E+00 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02003E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02005E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.11259E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.11461E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.22295E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.22006E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.59148E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.59690E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41127E-03 0.00554  2.63232E-04 0.01917  8.11796E-04 0.01177  6.00328E-04 0.01284  1.36297E-03 0.00842  3.08912E-04 0.01830  6.40369E-05 0.04011 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.95620E-01 0.01439  1.21557E-02 0.00614  3.22589E-02 0.00013  1.06446E-01 0.00058  3.00021E-01 0.00030  1.23378E+00 0.00526  4.40099E+00 0.03557 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.37282E-03 0.00754  2.65997E-04 0.02692  7.98773E-04 0.01637  5.90328E-04 0.01847  1.35395E-03 0.01214  3.00478E-04 0.02596  6.32915E-05 0.05537 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.96121E-01 0.02029  1.24765E-02 3.5E-05  3.22592E-02 0.00016  1.06390E-01 0.00071  3.00014E-01 0.00041  1.25708E+00 0.00090  7.30846E+00 0.01901 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.04942E-07 0.00114  9.04533E-07 0.00114  1.03032E-06 0.01573 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.23091E-07 0.00106  9.22674E-07 0.00106  1.05103E-06 0.01572 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.35415E-03 0.00885  2.58741E-04 0.03101  8.10231E-04 0.01876  5.76836E-04 0.02103  1.35399E-03 0.01366  2.93816E-04 0.02926  6.05325E-05 0.06764 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.84001E-01 0.02451  1.24754E-02 5.3E-05  3.22533E-02 0.00023  1.06293E-01 0.00098  3.00085E-01 0.00053  1.26020E+00 0.00125  7.05761E+00 0.03012 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.01728E-07 0.01299  8.01400E-07 0.01299  8.04272E-07 0.04561 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.18206E-07 0.01299  8.17874E-07 0.01299  8.20271E-07 0.04557 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80109E-03 0.03654  2.28112E-04 0.13878  6.31436E-04 0.07640  5.14845E-04 0.08088  1.12383E-03 0.05563  2.40453E-04 0.11567  6.24158E-05 0.23815 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.03523E-01 0.09841  1.24776E-02 0.00010  3.22689E-02 0.00047  1.06361E-01 0.00269  3.00024E-01 0.00161  1.25578E+00 0.00337  6.85702E+00 0.09739 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.79912E-03 0.03588  2.24872E-04 0.13362  6.22188E-04 0.07492  5.19264E-04 0.07848  1.12532E-03 0.05455  2.41691E-04 0.11411  6.57781E-05 0.23602 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.00263E-01 0.09707  1.24778E-02 9.9E-05  3.22690E-02 0.00046  1.06353E-01 0.00268  2.99977E-01 0.00160  1.25609E+00 0.00339  6.85702E+00 0.09739 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.53308E+03 0.03489 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.99895E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.17946E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30213E-03 0.00568 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.67060E+03 0.00573 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17660E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75145E-05 0.00992  6.75233E-05 0.00991  5.24439E-07 0.41757 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.02899E-05 0.02600  6.03323E-05 0.02599  2.54175E-07 0.49968 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.32461E-04 0.02046  2.32651E-04 0.02040  1.59935E-04 0.40817 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64225E+01 0.01136 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.70696E+01 0.00024  4.13824E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 12:32:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03659E+00  9.99982E-01  9.99536E-01  9.99135E-01  9.96883E-01  1.00018E+00  1.00597E+00  9.90022E-01  1.00684E+00  9.88239E-01  9.93841E-01  9.93190E-01  1.00730E+00  9.88147E-01  9.96734E-01  9.95419E-01  1.00795E+00  9.94036E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.26199E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77380E-01 6.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.13252E-01 0.00015  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.28472E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75793E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.69251E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.69251E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.59349E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.68964E-02 0.00168  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428895 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42885E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42885E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.17725E+03 ;
RUNNING_TIME              (idx, 1)        =  1.19353E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.62167E-01  2.68500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.17896E+02  1.36016E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.61117E-01  5.08667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.98900E-01  2.18167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.19345E+02  1.86690E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86360 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98183E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90504E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.88207E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73717E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.15371E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.98210E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36698E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86847E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56921E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.52300E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.00316E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.34378E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.29801E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.14488E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.06659E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.86129E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.39669E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35476E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.58183E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.62926E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.77761E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.25808E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96731E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.50028E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.05782E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61748E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30380E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.26557E-02 -4.87519E+27  2.20061E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00891E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.43308E+18 0.00419  1.53014E-02 0.00413 ];
U233_FISS                 (idx, [1:   4]) = [  8.31639E+19 0.00053  8.88059E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  5.92634E+18 0.00196  6.32861E-02 0.00192 ];
U238_FISS                 (idx, [1:   4]) = [  4.60089E+13 0.70679  4.89000E-07 0.70682 ];
PU239_FISS                (idx, [1:   4]) = [  7.85195E+16 0.01713  8.38452E-04 0.01713 ];
PU240_FISS                (idx, [1:   4]) = [  3.54704E+15 0.08118  3.78611E-05 0.08124 ];
PU241_FISS                (idx, [1:   4]) = [  3.90135E+15 0.07633  4.16627E-05 0.07635 ];
TH232_CAPT                (idx, [1:   4]) = [  9.55359E+19 0.00054  6.93924E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23261E+19 0.00139  8.95323E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  2.13474E+18 0.00322  1.55065E-02 0.00321 ];
U238_CAPT                 (idx, [1:   4]) = [  5.99970E+14 0.19261  4.35902E-06 0.19261 ];
PU239_CAPT                (idx, [1:   4]) = [  3.80456E+16 0.02480  2.76395E-04 0.02481 ];
PU240_CAPT                (idx, [1:   4]) = [  1.00076E+16 0.04835  7.27437E-05 0.04835 ];
PU241_CAPT                (idx, [1:   4]) = [  9.94562E+14 0.15158  7.21514E-06 0.15143 ];
SM149_CAPT                (idx, [1:   4]) = [  1.94074E+17 0.01118  1.40992E-03 0.01120 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001948 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10834E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001948 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5952554 5.95813E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4049227 4.05278E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 167 1.66887E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001948 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.16650E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35963E+20 4.0E-06  2.35963E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36761E+19 9.8E-08  9.36761E+19 9.8E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37584E+20 0.00023  1.27044E+20 0.00017  1.05402E+19 0.00195 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31260E+20 0.00013  2.20720E+20 9.7E-05  1.05402E+19 0.00195 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.31069E+20 0.00032  2.31069E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44639E+22 0.00022  3.32374E+22 0.00020  1.22644E+21 0.00193 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.85960E+15 0.08211 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.31264E+20 0.00013 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08541E+22 0.00028 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.18698E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.18698E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.72970E-01 0.05148 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24752E-01 0.01150 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.11022E-04 0.02140 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.95068E+03 0.01687 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99983E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.36219E-01 0.04381 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.36212E-01 0.04381 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51892E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99886E+02 9.8E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02051E+00 0.00043  2.07631E-02 0.00041  6.98607E-05 0.00833 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02148E+00 0.00013 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02125E+00 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02148E+00 0.00013 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02149E+00 0.00013 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.08916E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.08826E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.25193E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.25262E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.64008E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.64667E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.40674E-03 0.00537  2.47783E-04 0.01972  7.93096E-04 0.01137  6.11532E-04 0.01226  1.39044E-03 0.00860  3.00100E-04 0.01869  6.37913E-05 0.03947 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.93799E-01 0.01540  1.21741E-02 0.00597  3.22503E-02 0.00013  1.06435E-01 0.00056  3.00468E-01 0.00030  1.22677E+00 0.00620  4.24038E+00 0.03619 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.37995E-03 0.00767  2.42455E-04 0.02775  7.70526E-04 0.01586  6.18313E-04 0.01780  1.39280E-03 0.01244  2.90560E-04 0.02777  6.53022E-05 0.05599 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.01707E-01 0.02293  1.24769E-02 3.4E-05  3.22508E-02 0.00017  1.06411E-01 0.00073  3.00314E-01 0.00041  1.25876E+00 0.00099  6.99883E+00 0.01988 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.93311E-07 0.00117  8.92841E-07 0.00116  1.02558E-06 0.01579 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.11505E-07 0.00107  9.11026E-07 0.00107  1.04658E-06 0.01578 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.34731E-03 0.00845  2.34242E-04 0.03340  7.96402E-04 0.01801  5.91590E-04 0.02015  1.37332E-03 0.01315  2.87061E-04 0.02972  6.46901E-05 0.06241 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.99959E-01 0.02542  1.24772E-02 4.2E-05  3.22442E-02 0.00022  1.06634E-01 0.00109  3.00335E-01 0.00052  1.25953E+00 0.00134  7.26065E+00 0.02762 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.87783E-07 0.01294  7.87354E-07 0.01295  7.94021E-07 0.05099 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.04112E-07 0.01293  8.03672E-07 0.01294  8.10576E-07 0.05096 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.27513E-03 0.03766  2.44858E-04 0.12411  7.85893E-04 0.07093  5.68010E-04 0.07825  1.36097E-03 0.05672  2.49680E-04 0.12097  6.57189E-05 0.20112 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.33254E-01 0.09916  1.24774E-02 0.00010  3.22478E-02 0.00066  1.06453E-01 0.00266  2.99731E-01 0.00154  1.25968E+00 0.00369  7.85783E+00 0.07134 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.24427E-03 0.03703  2.43769E-04 0.12161  7.77562E-04 0.07062  5.61709E-04 0.07690  1.35247E-03 0.05565  2.46005E-04 0.11812  6.27558E-05 0.20570 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.29469E-01 0.09882  1.24773E-02 0.00011  3.22467E-02 0.00065  1.06469E-01 0.00267  2.99770E-01 0.00154  1.25982E+00 0.00370  7.85783E+00 0.07134 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.19612E+03 0.03594 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.88005E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.06094E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.41859E-03 0.00523 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.85152E+03 0.00532 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.13368E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67361E-05 0.01035  6.67901E-05 0.01036  6.04736E-07 0.38403 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.71649E-05 0.02722  5.71377E-05 0.02718  6.05305E-07 0.66327 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.20315E-04 0.02100  2.20368E-04 0.02107  1.96342E-04 0.37985 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60431E+01 0.01205 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.69251E+01 0.00023  4.11536E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 12:46:32 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02622E+00  9.97525E-01  9.97719E-01  9.98234E-01  9.97719E-01  1.00242E+00  1.00811E+00  9.95855E-01  9.94894E-01  1.00768E+00  9.88318E-01  9.89165E-01  9.99332E-01  9.96930E-01  9.95032E-01  9.94769E-01  1.00007E+00  1.01000E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.26491E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77351E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.13208E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.28449E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75846E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.68223E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.68223E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.57351E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.67992E-02 0.00168  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429138 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42891E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42891E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.31280E+03 ;
RUNNING_TIME              (idx, 1)        =  1.33043E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.88783E-01  2.66167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.31476E+02  1.35805E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.10467E-01  4.93500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.20917E-01  2.19000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.33035E+02  1.87745E+02 ];
CPU_USAGE                 (idx, 1)        = 9.86753 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98209E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90930E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 186 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.87199E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73536E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.28583E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96420E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35474E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87627E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56859E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.74099E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17923E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.48467E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.48061E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21999E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.22401E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.24095E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38267E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34304E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56639E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.64515E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.15540E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24322E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.98110E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.84143E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.02481E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61765E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96390E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.79320E-02 -6.01057E+27  2.21196E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00486E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.41119E+18 0.00397  1.50616E-02 0.00392 ];
U233_FISS                 (idx, [1:   4]) = [  8.23872E+19 0.00051  8.79371E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  6.41622E+18 0.00202  6.84838E-02 0.00195 ];
U238_FISS                 (idx, [1:   4]) = [  6.95153E+13 0.57653  7.41405E-07 0.57655 ];
PU239_FISS                (idx, [1:   4]) = [  1.32501E+17 0.01350  1.41424E-03 0.01350 ];
PU240_FISS                (idx, [1:   4]) = [  6.02496E+15 0.06390  6.43521E-05 0.06394 ];
PU241_FISS                (idx, [1:   4]) = [  9.81544E+15 0.04982  1.04753E-04 0.04980 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44879E+19 0.00052  6.86410E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22398E+19 0.00139  8.89185E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  2.30570E+18 0.00315  1.67519E-02 0.00318 ];
U238_CAPT                 (idx, [1:   4]) = [  6.23932E+14 0.18887  4.52441E-06 0.18887 ];
PU239_CAPT                (idx, [1:   4]) = [  6.61968E+16 0.01785  4.80913E-04 0.01785 ];
PU240_CAPT                (idx, [1:   4]) = [  1.91387E+16 0.03574  1.39109E-04 0.03574 ];
PU241_CAPT                (idx, [1:   4]) = [  1.77886E+15 0.11228  1.29198E-05 0.11222 ];
SM149_CAPT                (idx, [1:   4]) = [  1.89764E+17 0.01130  1.37891E-03 0.01133 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002343 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10714E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002343 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5951288 5.95668E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4050861 4.05420E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 194 1.94070E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002343 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -8.94070E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35987E+20 4.0E-06  2.35987E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36628E+19 9.7E-08  9.36628E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37614E+20 0.00024  1.27112E+20 0.00018  1.05022E+19 0.00198 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31277E+20 0.00014  2.20775E+20 0.00011  1.05022E+19 0.00198 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.31093E+20 0.00033  2.31093E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44215E+22 0.00023  3.31991E+22 0.00021  1.22248E+21 0.00202 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.49317E+15 0.08419 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.31281E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08316E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.17195E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.17195E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.66507E-01 0.05025 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.37141E-01 0.01151 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.14547E-04 0.02170 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.00452E+03 0.01594 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99981E-01 1.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.61170E-01 0.04170 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.61162E-01 0.04170 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51953E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99914E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02136E+00 0.00040  2.07761E-02 0.00039  7.02062E-05 0.00877 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02150E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02125E+00 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02150E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02152E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.06998E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06908E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.27621E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.27685E-02 0.00086 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.68281E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.68274E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.46742E-03 0.00543  2.53278E-04 0.01958  8.30449E-04 0.01113  6.25814E-04 0.01265  1.37337E-03 0.00843  3.16917E-04 0.01794  6.75897E-05 0.03911 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.96436E-01 0.01469  1.21380E-02 0.00632  3.22309E-02 0.00013  1.06575E-01 0.00055  3.00550E-01 0.00032  1.25143E+00 0.00301  4.26390E+00 0.03533 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.39170E-03 0.00782  2.40612E-04 0.02678  8.16116E-04 0.01607  6.13755E-04 0.01847  1.34670E-03 0.01214  3.09032E-04 0.02569  6.54830E-05 0.05949 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.91165E-01 0.02076  1.24764E-02 3.8E-05  3.22340E-02 0.00017  1.06559E-01 0.00072  3.00434E-01 0.00044  1.26010E+00 0.00114  6.91085E+00 0.01968 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.84765E-07 0.00113  8.84373E-07 0.00113  9.96363E-07 0.01560 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.03556E-07 0.00104  9.03156E-07 0.00104  1.01761E-06 0.01561 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.37596E-03 0.00904  2.46119E-04 0.03239  8.17656E-04 0.01795  6.04507E-04 0.02052  1.32951E-03 0.01340  3.15411E-04 0.02736  6.27638E-05 0.06531 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.88482E-01 0.02454  1.24769E-02 4.8E-05  3.22285E-02 0.00022  1.06473E-01 0.00101  3.00629E-01 0.00056  1.25689E+00 0.00146  6.51099E+00 0.03067 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.81962E-07 0.01296  7.81508E-07 0.01296  7.88312E-07 0.05038 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.98916E-07 0.01295  7.98452E-07 0.01295  8.05440E-07 0.05031 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01675E-03 0.03686  1.73353E-04 0.14845  6.83890E-04 0.07368  5.44522E-04 0.08355  1.29826E-03 0.05444  2.60353E-04 0.10921  5.63713E-05 0.23554 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.94669E-01 0.07433  1.24779E-02 0.00012  3.22316E-02 0.00069  1.06677E-01 0.00294  3.01269E-01 0.00169  1.25736E+00 0.00330  6.96848E+00 0.08931 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99079E-03 0.03648  1.80343E-04 0.14744  6.84478E-04 0.07309  5.30794E-04 0.08116  1.27102E-03 0.05364  2.63970E-04 0.10866  6.01917E-05 0.23555 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.01671E-01 0.07688  1.24778E-02 0.00012  3.22317E-02 0.00069  1.06669E-01 0.00294  3.01266E-01 0.00169  1.25759E+00 0.00331  6.96848E+00 0.08931 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88859E+03 0.03481 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.78832E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.97494E-07 0.00048 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42916E-03 0.00505 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90363E+03 0.00512 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.11885E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77622E-05 0.00975  6.77421E-05 0.00977  7.41980E-07 0.38944 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.77842E-05 0.02552  5.76240E-05 0.02561  9.73408E-07 0.58374 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.25134E-04 0.02135  2.25223E-04 0.02142  1.93945E-04 0.37780 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63713E+01 0.01127 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.68223E+01 0.00021  4.09691E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 13:00:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02775E+00  9.96463E-01  1.01015E+00  9.95445E-01  1.00229E+00  9.95285E-01  9.89419E-01  1.00627E+00  9.90963E-01  1.00655E+00  9.92015E-01  9.86183E-01  1.00314E+00  1.00124E+00  1.00261E+00  1.00030E+00  9.88378E-01  1.00553E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.27316E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77268E-01 6.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.12985E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.28284E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76352E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.66871E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.66871E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.55299E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.73623E-02 0.00165  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429032 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42902E+04 0.00051 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42902E+04 0.00051 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.44893E+03 ;
RUNNING_TIME              (idx, 1)        =  1.46790E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.17817E-01  2.90333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.45113E+02  1.36367E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.57717E-01  4.72500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.44367E-01  2.33667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.46782E+02  1.87753E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87079 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98259E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91264E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.89405E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73994E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.15404E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93591E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33928E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92667E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57478E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.05186E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75630E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.03196E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03705E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44668E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.71137E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.36283E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37374E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33351E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55506E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.68126E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.29482E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24072E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02533E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.39464E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01113E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.62223E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60434E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -4.90470E-02 -1.05542E+28  2.25740E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00396E+00 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.38937E+18 0.00416  1.48392E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  8.04072E+19 0.00050  8.58866E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  7.29284E+18 0.00172  7.78999E-02 0.00168 ];
U238_FISS                 (idx, [1:   4]) = [  4.62714E+13 0.70663  4.94786E-07 0.70660 ];
PU239_FISS                (idx, [1:   4]) = [  3.87830E+17 0.00779  4.14254E-03 0.00778 ];
PU240_FISS                (idx, [1:   4]) = [  2.77044E+16 0.02897  2.96057E-04 0.02898 ];
PU241_FISS                (idx, [1:   4]) = [  5.11247E+16 0.02056  5.46058E-04 0.02053 ];
TH232_CAPT                (idx, [1:   4]) = [  9.24722E+19 0.00051  6.68243E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.19195E+19 0.00139  8.61361E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  2.60265E+18 0.00281  1.88092E-02 0.00281 ];
U238_CAPT                 (idx, [1:   4]) = [  1.57445E+15 0.12621  1.13785E-05 0.12614 ];
PU239_CAPT                (idx, [1:   4]) = [  1.88279E+17 0.01095  1.36077E-03 0.01096 ];
PU240_CAPT                (idx, [1:   4]) = [  8.55231E+16 0.01687  6.18162E-04 0.01688 ];
PU241_CAPT                (idx, [1:   4]) = [  1.13523E+16 0.04549  8.20539E-05 0.04553 ];
SM149_CAPT                (idx, [1:   4]) = [  1.94496E+17 0.01075  1.40544E-03 0.01072 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003142 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11556E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003142 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5966148 5.97118E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4036795 4.03978E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 199 1.98690E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003142 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.31323E-09 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36170E+20 4.1E-06  2.36170E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36213E+19 9.1E-08  9.36213E+19 9.1E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.38463E+20 0.00024  1.27856E+20 0.00017  1.06063E+19 0.00198 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.32084E+20 0.00014  2.21478E+20 0.00010  1.06063E+19 0.00198 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.31748E+20 0.00033  2.31748E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.45012E+22 0.00023  3.32656E+22 0.00021  1.23554E+21 0.00199 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.60399E+15 0.08256 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.32089E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08310E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  8.11369E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.11369E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.39483E-01 0.05294 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33426E-01 0.01091 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.16791E-04 0.02169 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.93021E+03 0.01710 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99980E-01 1.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.27053E-01 0.04458 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.27045E-01 0.04458 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52262E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00003E+02 9.1E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.01894E+00 0.00036  2.07264E-02 0.00036  7.10899E-05 0.00855 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.01872E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.01916E+00 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.01872E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01874E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02868E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02824E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32998E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33009E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.77257E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.76942E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.52710E-03 0.00545  2.59670E-04 0.01928  8.18431E-04 0.01121  6.34639E-04 0.01182  1.41939E-03 0.00837  3.25505E-04 0.01762  6.94712E-05 0.03737 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08910E-01 0.01486  1.21046E-02 0.00665  3.22042E-02 0.00015  1.06647E-01 0.00055  3.01049E-01 0.00029  1.25087E+00 0.00367  4.71418E+00 0.03280 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43928E-03 0.00713  2.49021E-04 0.02833  8.00779E-04 0.01539  6.13802E-04 0.01696  1.38376E-03 0.01151  3.20382E-04 0.02502  7.15382E-05 0.05489 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.18877E-01 0.02199  1.24780E-02 7.3E-05  3.21968E-02 0.00019  1.06779E-01 0.00076  3.01034E-01 0.00040  1.26278E+00 0.00127  7.34547E+00 0.01768 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.70089E-07 0.00108  8.69633E-07 0.00108  1.00616E-06 0.03019 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.86497E-07 0.00103  8.86033E-07 0.00103  1.02459E-06 0.02995 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40515E-03 0.00877  2.52204E-04 0.03326  7.85975E-04 0.01828  6.15832E-04 0.01952  1.37180E-03 0.01377  3.13090E-04 0.02897  6.62503E-05 0.06181 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04593E-01 0.02352  1.24777E-02 7.1E-05  3.21992E-02 0.00029  1.06729E-01 0.00105  3.00997E-01 0.00049  1.26042E+00 0.00181  7.27698E+00 0.02679 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.68151E-07 0.01295  7.67669E-07 0.01295  8.03977E-07 0.04182 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.82827E-07 0.01294  7.82335E-07 0.01295  8.19385E-07 0.04183 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.88343E-03 0.03684  1.95695E-04 0.12238  6.32965E-04 0.07939  5.51900E-04 0.07908  1.15045E-03 0.05811  2.88633E-04 0.11263  6.37873E-05 0.21227 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.72035E-01 0.09197  1.24771E-02 0.00012  3.22044E-02 0.00059  1.06870E-01 0.00285  3.01203E-01 0.00166  1.25945E+00 0.00476  7.00543E+00 0.07451 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87659E-03 0.03581  1.99250E-04 0.12226  6.26492E-04 0.07719  5.49591E-04 0.07708  1.15564E-03 0.05683  2.84655E-04 0.10968  6.09607E-05 0.21186 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.70614E-01 0.09083  1.24771E-02 0.00011  3.22039E-02 0.00059  1.06855E-01 0.00284  3.01201E-01 0.00166  1.25944E+00 0.00476  6.99348E+00 0.07453 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.78209E+03 0.03524 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.64666E-07 0.00060 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.80964E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37697E-03 0.00514 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90593E+03 0.00512 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.09542E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.55044E-05 0.01025  6.54220E-05 0.01042  8.55941E-07 0.35614 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.74723E-05 0.02523  5.71275E-05 0.02514  1.03074E-06 0.42550 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.27403E-04 0.02120  2.27401E-04 0.02125  2.35470E-04 0.35586 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62124E+01 0.01124 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.66871E+01 0.00022  4.05645E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 13:14:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02542E+00  9.90698E-01  9.91247E-01  1.00011E+00  1.00219E+00  9.94655E-01  9.91236E-01  1.00338E+00  1.00613E+00  9.97719E-01  9.86171E-01  1.00347E+00  1.00577E+00  1.00014E+00  9.88800E-01  1.01080E+00  9.97136E-01  1.00492E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.28693E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77131E-01 6.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.11733E-01 0.00017  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.27150E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.77843E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.65943E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.65942E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.58321E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.85128E-02 0.00169  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428846 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42887E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42887E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.58514E+03 ;
RUNNING_TIME              (idx, 1)        =  1.60544E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.43300E-01  2.54833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.58756E+02  1.36432E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.06517E-01  4.88000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.66067E-01  2.16167E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.60536E+02  1.87958E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87355 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98246E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91542E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.89023E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73413E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.04685E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90746E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32736E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95134E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57020E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57731E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.37978E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.40368E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13648E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69173E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23452E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57680E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34471E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31038E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52407E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.70804E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.60217E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21553E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99741E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.10306E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.97320E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.62782E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32054E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.04547E-01 -2.24970E+28  2.37683E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00535E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.36849E+18 0.00413  1.46247E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  7.91668E+19 0.00053  8.46043E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  7.45649E+18 0.00173  7.96859E-02 0.00166 ];
U238_FISS                 (idx, [1:   4]) = [  1.62850E+14 0.37637  1.74266E-06 0.37640 ];
PU239_FISS                (idx, [1:   4]) = [  6.99362E+17 0.00561  7.47445E-03 0.00560 ];
PU240_FISS                (idx, [1:   4]) = [  8.05039E+16 0.01632  8.60166E-04 0.01629 ];
PU241_FISS                (idx, [1:   4]) = [  1.62025E+17 0.01270  1.73156E-03 0.01270 ];
TH232_CAPT                (idx, [1:   4]) = [  9.13039E+19 0.00054  6.55825E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.16923E+19 0.00135  8.39884E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  2.67777E+18 0.00289  1.92354E-02 0.00289 ];
U238_CAPT                 (idx, [1:   4]) = [  2.90579E+15 0.09146  2.08539E-05 0.09138 ];
PU239_CAPT                (idx, [1:   4]) = [  3.43130E+17 0.00812  2.46449E-03 0.00809 ];
PU240_CAPT                (idx, [1:   4]) = [  2.31308E+17 0.01005  1.66138E-03 0.01004 ];
PU241_CAPT                (idx, [1:   4]) = [  3.54997E+16 0.02586  2.54936E-04 0.02585 ];
SM149_CAPT                (idx, [1:   4]) = [  1.93602E+17 0.01144  1.39071E-03 0.01144 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002108 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08199E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002108 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5981379 5.98674E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4020550 4.02390E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 179 1.79044E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002108 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.11759E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36474E+20 4.2E-06  2.36474E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35818E+19 9.6E-08  9.35818E+19 9.6E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.39194E+20 0.00026  1.28446E+20 0.00018  1.07484E+19 0.00208 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.32776E+20 0.00015  2.22028E+20 0.00011  1.07484E+19 0.00208 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.32546E+20 0.00034  2.32546E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47677E+22 0.00024  3.35211E+22 0.00021  1.24661E+21 0.00207 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.17786E+15 0.09094 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.32780E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08462E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  7.99717E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.99717E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.90713E-01 0.05212 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27450E-01 0.01051 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.39118E-04 0.01973 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.07747E+03 0.01523 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 1.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.44485E-01 0.04292 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.44478E-01 0.04292 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52692E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00087E+02 9.6E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.01655E+00 0.00041  2.06814E-02 0.00039  7.02282E-05 0.00873 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.01699E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.01697E+00 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.01699E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01700E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00068E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00083E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36770E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36706E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83248E-01 0.00137 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.82807E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49239E-03 0.00540  2.44706E-04 0.02039  8.23967E-04 0.01088  6.15174E-04 0.01220  1.41444E-03 0.00858  3.20979E-04 0.01708  7.31245E-05 0.03620 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.17726E-01 0.01468  1.21209E-02 0.00649  3.21833E-02 0.00017  1.06510E-01 0.00053  3.01178E-01 0.00030  1.25114E+00 0.00367  4.77903E+00 0.03181 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43793E-03 0.00761  2.41481E-04 0.02886  8.21882E-04 0.01501  5.92640E-04 0.01785  1.38768E-03 0.01195  3.24541E-04 0.02599  6.97138E-05 0.05104 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.09734E-01 0.01923  1.24775E-02 5.8E-05  3.21872E-02 0.00022  1.06511E-01 0.00071  3.01271E-01 0.00043  1.26407E+00 0.00124  7.17208E+00 0.01760 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.63949E-07 0.00111  8.63591E-07 0.00111  9.66480E-07 0.01488 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.78151E-07 0.00103  8.77788E-07 0.00104  9.82013E-07 0.01482 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.38710E-03 0.00881  2.31518E-04 0.03215  7.96251E-04 0.01842  6.01094E-04 0.02051  1.37984E-03 0.01378  3.09673E-04 0.03002  6.87220E-05 0.06028 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.12764E-01 0.02374  1.24785E-02 9.2E-05  3.21825E-02 0.00028  1.06580E-01 0.00098  3.01352E-01 0.00052  1.25901E+00 0.00190  7.32111E+00 0.02536 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.61421E-07 0.01306  7.61167E-07 0.01307  7.58772E-07 0.04821 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.74343E-07 0.01306  7.74089E-07 0.01306  7.71097E-07 0.04811 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.27587E-03 0.03792  2.05169E-04 0.12757  7.51836E-04 0.07485  6.35101E-04 0.08282  1.32275E-03 0.05692  2.97080E-04 0.12398  6.39268E-05 0.23975 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.33811E-01 0.10158  1.24744E-02 0.00017  3.21934E-02 0.00082  1.06860E-01 0.00275  3.01782E-01 0.00164  1.26308E+00 0.00465  7.97851E+00 0.05326 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.30804E-03 0.03775  2.11358E-04 0.12887  7.58919E-04 0.07511  6.40073E-04 0.08102  1.34503E-03 0.05586  2.89557E-04 0.12213  6.31002E-05 0.24139 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.31596E-01 0.10172  1.24743E-02 0.00017  3.21938E-02 0.00083  1.06864E-01 0.00274  3.01867E-01 0.00164  1.26327E+00 0.00466  7.97851E+00 0.05326 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.31724E+03 0.03576 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.58546E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.72663E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47426E-03 0.00580 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.04756E+03 0.00583 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.11976E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69737E-05 0.00927  6.69443E-05 0.00929  9.01746E-07 0.33445 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.07227E-05 0.02423  6.05494E-05 0.02432  1.33783E-06 0.42318 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.49091E-04 0.01949  2.49065E-04 0.01957  2.49484E-04 0.33257 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59772E+01 0.01144 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.65942E+01 0.00023  4.02685E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 13:27:53 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01804E+00  9.99457E-01  1.00901E+00  9.88319E-01  1.00285E+00  9.99217E-01  1.00439E+00  1.00817E+00  9.89966E-01  9.93305E-01  1.00617E+00  1.00146E+00  1.00007E+00  1.00640E+00  9.95489E-01  9.96507E-01  9.87129E-01  9.94048E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.30233E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.76977E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.10339E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.25886E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.79253E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.66425E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.66425E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.64838E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.04627E-02 0.00165  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428660 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42881E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42881E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.72217E+03 ;
RUNNING_TIME              (idx, 1)        =  1.74393E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.73767E-01  3.04667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.72481E+02  1.37246E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.58383E-01  5.18667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.91300E-01  2.51333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.74385E+02  1.88110E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87522 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98261E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91725E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.89394E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73295E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.76447E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90266E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32578E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95973E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56906E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65919E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55081E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47816E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.15261E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76280E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.38884E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.92203E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33790E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30533E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51705E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.71994E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00720E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21057E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.98765E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27056E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.96892E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.62782E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98065E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61179E-01 -3.46834E+28  2.49869E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00164E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.36572E+18 0.00407  1.45893E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  7.91195E+19 0.00053  8.45240E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.42474E+18 0.00170  7.93236E-02 0.00168 ];
U238_FISS                 (idx, [1:   4]) = [  1.16047E+14 0.44600  1.24118E-06 0.44594 ];
PU239_FISS                (idx, [1:   4]) = [  7.31655E+17 0.00558  7.81702E-03 0.00558 ];
PU240_FISS                (idx, [1:   4]) = [  9.43774E+16 0.01558  1.00836E-03 0.01559 ];
PU241_FISS                (idx, [1:   4]) = [  1.85334E+17 0.01121  1.97932E-03 0.01118 ];
TH232_CAPT                (idx, [1:   4]) = [  9.08821E+19 0.00053  6.52926E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.16882E+19 0.00139  8.39733E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  2.67610E+18 0.00295  1.92269E-02 0.00294 ];
U238_CAPT                 (idx, [1:   4]) = [  3.50741E+15 0.08302  2.52051E-05 0.08300 ];
PU239_CAPT                (idx, [1:   4]) = [  3.59513E+17 0.00847  2.58310E-03 0.00848 ];
PU240_CAPT                (idx, [1:   4]) = [  2.72191E+17 0.00881  1.95576E-03 0.00882 ];
PU241_CAPT                (idx, [1:   4]) = [  3.93409E+16 0.02463  2.82614E-04 0.02463 ];
SM149_CAPT                (idx, [1:   4]) = [  1.98740E+17 0.01080  1.42789E-03 0.01079 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001640 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10084E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001640 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5979775 5.98553E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4021694 4.02531E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 171 1.70959E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001640 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.34110E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36527E+20 4.3E-06  2.36527E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35764E+19 1.0E-07  9.35764E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.39286E+20 0.00023  1.28350E+20 0.00017  1.09367E+19 0.00194 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.32863E+20 0.00014  2.21926E+20 9.9E-05  1.09367E+19 0.00194 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.32545E+20 0.00032  2.32545E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50077E+22 0.00022  3.37484E+22 0.00020  1.25930E+21 0.00197 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.98085E+15 0.08051 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.32867E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08577E+22 0.00029 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  7.89171E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.89171E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.63845E-01 0.04491 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.22300E-01 0.01092 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.63658E-04 0.02048 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.82981E+03 0.01649 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99983E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.24718E-01 0.03666 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.24708E-01 0.03666 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52764E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00099E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.01741E+00 0.00041  2.06943E-02 0.00040  6.96274E-05 0.00850 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.01684E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.01719E+00 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.01684E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01686E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00286E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00195E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36469E-02 0.00127 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36553E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83385E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83224E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49532E-03 0.00521  2.46706E-04 0.02101  8.16909E-04 0.01075  6.33403E-04 0.01289  1.41346E-03 0.00826  3.17447E-04 0.01774  6.73950E-05 0.03737 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.02587E-01 0.01502  1.21580E-02 0.00615  3.21892E-02 0.00017  1.06735E-01 0.00057  3.01326E-01 0.00030  1.23901E+00 0.00473  4.46711E+00 0.03402 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.39776E-03 0.00762  2.29807E-04 0.02903  7.82724E-04 0.01582  6.22013E-04 0.01881  1.38548E-03 0.01206  3.10742E-04 0.02626  6.69894E-05 0.05451 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02542E-01 0.02008  1.24794E-02 9.6E-05  3.21940E-02 0.00023  1.06748E-01 0.00074  3.01301E-01 0.00042  1.25620E+00 0.00155  6.94210E+00 0.01914 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.71368E-07 0.00116  8.70966E-07 0.00116  9.87380E-07 0.01625 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.86440E-07 0.00110  8.86030E-07 0.00110  1.00467E-06 0.01627 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.34703E-03 0.00867  2.37775E-04 0.03435  7.80450E-04 0.01833  6.00304E-04 0.02071  1.34891E-03 0.01398  3.12740E-04 0.02933  6.68530E-05 0.06149 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.09150E-01 0.02558  1.24798E-02 0.00012  3.21856E-02 0.00028  1.06695E-01 0.00099  3.01315E-01 0.00056  1.25702E+00 0.00193  6.87514E+00 0.02902 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.70112E-07 0.01333  7.69624E-07 0.01334  7.97110E-07 0.04729 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.83644E-07 0.01333  7.83147E-07 0.01333  8.11555E-07 0.04734 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.12986E-03 0.03767  2.16605E-04 0.13805  6.86078E-04 0.07515  5.73587E-04 0.09633  1.31019E-03 0.05483  2.72047E-04 0.12042  7.13486E-05 0.27178 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.50696E-01 0.10572  1.24785E-02 0.00010  3.22060E-02 0.00080  1.06740E-01 0.00272  3.01877E-01 0.00171  1.26498E+00 0.00557  7.59700E+00 0.07543 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.11365E-03 0.03710  2.17030E-04 0.13546  6.99941E-04 0.07488  5.64604E-04 0.09333  1.29857E-03 0.05361  2.62977E-04 0.11954  7.05237E-05 0.27660 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.47121E-01 0.10537  1.24785E-02 0.00010  3.22073E-02 0.00080  1.06746E-01 0.00270  3.01888E-01 0.00170  1.26505E+00 0.00557  7.60765E+00 0.07527 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.10879E+03 0.03565 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.64033E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.78973E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34637E-03 0.00596 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.87428E+03 0.00601 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17796E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.74375E-05 0.00837  6.74254E-05 0.00836  8.25974E-07 0.37237 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.95651E-05 0.02170  5.95722E-05 0.02175  8.29759E-07 0.47390 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.75763E-04 0.01999  2.75917E-04 0.02001  2.42943E-04 0.35284 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60413E+01 0.01125 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.66425E+01 0.00023  4.02754E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 29])  = 'msfr_mix1_benchmark_burn_rho5' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 10:33:30 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 13:39:01 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644327210158 ;
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
SPECTRUM_COLLAPSE         (idx, 1)        = 1 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 7 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03063E+00  9.93534E-01  9.95066E-01  1.00019E+00  9.92230E-01  9.99137E-01  1.00341E+00  1.01128E+00  1.00266E+00  9.97147E-01  9.91053E-01  1.00639E+00  9.89543E-01  1.01042E+00  9.90847E-01  9.91327E-01  9.95924E-01  9.99217E-01  ];
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

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 4.7E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  2.32104E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.76790E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.09285E-01 0.00017  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.24977E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.80357E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.67060E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.67059E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.70160E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  7.22937E-02 0.00162  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428742 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42885E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42885E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.83198E+03 ;
RUNNING_TIME              (idx, 1)        =  1.85521E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  5.61550E-01  5.61550E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.05700E-01  3.19333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.83480E+02  1.09992E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.08583E-01  5.02000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.17317E-01  2.59333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.85513E+02  1.85513E+02 ];
CPU_USAGE                 (idx, 1)        = 9.87481 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97942E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.91734E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7451.41;
MEMSIZE                   (idx, 1)        = 7278.04;
XS_MEMSIZE                (idx, 1)        = 6986.13;
MAT_MEMSIZE               (idx, 1)        = 162.18;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 173.37;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 278726 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.89454E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73238E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.97416E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90124E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32535E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96171E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56851E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.66930E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.59854E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.48603E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.15535E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78325E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43347E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.01990E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33539E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30342E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51443E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.71755E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13088E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20844E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.98376E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.30893E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.96666E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.62959E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64076E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.18535E-01 -4.70256E+28  2.62211E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00309E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.36791E+18 0.00408  1.46242E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  7.90055E+19 0.00050  8.44667E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.43025E+18 0.00180  7.94388E-02 0.00174 ];
U238_FISS                 (idx, [1:   4]) = [  2.08773E+14 0.33144  2.22720E-06 0.33145 ];
PU239_FISS                (idx, [1:   4]) = [  7.42306E+17 0.00592  7.93622E-03 0.00591 ];
PU240_FISS                (idx, [1:   4]) = [  9.31914E+16 0.01598  9.96133E-04 0.01596 ];
PU241_FISS                (idx, [1:   4]) = [  1.86758E+17 0.01084  1.99666E-03 0.01083 ];
TH232_CAPT                (idx, [1:   4]) = [  9.09015E+19 0.00052  6.51570E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.16892E+19 0.00144  8.37864E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  2.66799E+18 0.00277  1.91246E-02 0.00276 ];
U238_CAPT                 (idx, [1:   4]) = [  3.65107E+15 0.07798  2.61781E-05 0.07799 ];
PU239_CAPT                (idx, [1:   4]) = [  3.64343E+17 0.00774  2.61123E-03 0.00771 ];
PU240_CAPT                (idx, [1:   4]) = [  2.74538E+17 0.00933  1.96829E-03 0.00936 ];
PU241_CAPT                (idx, [1:   4]) = [  4.19621E+16 0.02360  3.00815E-04 0.02362 ];
SM149_CAPT                (idx, [1:   4]) = [  1.97807E+17 0.01050  1.41781E-03 0.01049 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001973 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08413E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001973 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5987332 5.99278E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4014459 4.01788E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 182 1.81947E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001973 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.72529E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61703E-02 1.9E-09  3.61703E-02 1.9E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36533E+20 4.2E-06  2.36533E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35758E+19 1.0E-07  9.35758E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.39484E+20 0.00024  1.28330E+20 0.00017  1.11545E+19 0.00195 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33060E+20 0.00014  2.21906E+20 9.9E-05  1.11545E+19 0.00195 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.32799E+20 0.00032  2.32799E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.52731E+22 0.00023  3.39932E+22 0.00020  1.27987E+21 0.00204 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.23959E+15 0.08255 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.33064E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08842E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29409E+04 ;
TOT_FMASS                 (idx, 1)        =  7.79312E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29409E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.79312E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.87301E-01 0.04450 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.18403E-01 0.01016 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.86876E-04 0.01930 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.84225E+03 0.01591 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 1.5E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.45436E-01 0.03511 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.45427E-01 0.03511 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52771E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00100E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.01555E+00 0.00039  2.06559E-02 0.00038  7.01615E-05 0.00854 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.01600E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.01611E+00 0.00031 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.01600E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01601E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00341E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00490E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36396E-02 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36148E-02 0.00084 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.82979E-01 0.00143 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.82938E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49900E-03 0.00548  2.45918E-04 0.01952  8.15195E-04 0.01134  6.39095E-04 0.01275  1.39739E-03 0.00851  3.26150E-04 0.01716  7.52501E-05 0.03791 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.15854E-01 0.01462  1.20720E-02 0.00697  3.21912E-02 0.00018  1.06748E-01 0.00056  3.01066E-01 0.00030  1.25144E+00 0.00338  4.49550E+00 0.03341 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42850E-03 0.00770  2.41364E-04 0.02963  7.81582E-04 0.01557  6.37457E-04 0.01806  1.37268E-03 0.01153  3.20734E-04 0.02586  7.46837E-05 0.05639 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.11432E-01 0.01957  1.24830E-02 0.00017  3.21867E-02 0.00022  1.06714E-01 0.00072  3.01049E-01 0.00042  1.25992E+00 0.00132  6.99035E+00 0.01842 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.78270E-07 0.00115  8.77844E-07 0.00116  1.00679E-06 0.01498 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.91837E-07 0.00109  8.91403E-07 0.00109  1.02271E-06 0.01498 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40219E-03 0.00857  2.35284E-04 0.03296  7.81440E-04 0.01856  6.30476E-04 0.02028  1.37217E-03 0.01340  3.11738E-04 0.02871  7.10802E-05 0.06057 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.20855E-01 0.02498  1.24819E-02 0.00019  3.21923E-02 0.00030  1.06758E-01 0.00098  3.01208E-01 0.00052  1.25808E+00 0.00176  7.34199E+00 0.02541 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.72958E-07 0.01297  7.72405E-07 0.01297  8.52822E-07 0.04391 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.85225E-07 0.01296  7.84664E-07 0.01296  8.66397E-07 0.04391 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.03630E-03 0.03661  2.52509E-04 0.13506  6.40639E-04 0.07498  5.39414E-04 0.08253  1.23299E-03 0.05749  3.18602E-04 0.12026  5.21468E-05 0.22246 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.91171E-01 0.09061  1.24933E-02 0.00089  3.22055E-02 0.00087  1.06085E-01 0.00214  3.00053E-01 0.00157  1.26325E+00 0.00465  6.25091E+00 0.09525 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03073E-03 0.03594  2.55964E-04 0.12990  6.37119E-04 0.07400  5.33192E-04 0.08050  1.22800E-03 0.05723  3.20919E-04 0.11731  5.55335E-05 0.21657 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.96628E-01 0.08936  1.24934E-02 0.00089  3.22058E-02 0.00087  1.06082E-01 0.00213  3.00088E-01 0.00157  1.26323E+00 0.00464  6.24834E+00 0.09520 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96190E+03 0.03464 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.72229E-07 0.00065 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.85692E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37127E-03 0.00567 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.86695E+03 0.00574 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.23409E-09 0.00087 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75796E-05 0.00850  6.75915E-05 0.00852  9.57375E-07 0.34094 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.00416E-05 0.02084  6.01343E-05 0.02092  7.27622E-07 0.39975 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.99290E-04 0.01876  2.99437E-04 0.01877  2.54031E-04 0.33265 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59808E+01 0.01060 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.67059E+01 0.00021  4.02712E+01 0.00032 ];

