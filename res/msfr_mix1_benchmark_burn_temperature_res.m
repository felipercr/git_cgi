
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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 04:34:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.37271E+00  1.01195E+00  9.81520E-01  9.81154E-01  9.71118E-01  9.66751E-01  9.94756E-01  9.74890E-01  9.83360E-01  9.87647E-01  9.64671E-01  9.94997E-01  9.95591E-01  1.02040E+00  9.42827E-01  9.85795E-01  8.83856E-01  9.86012E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.18559E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78144E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.18135E-01 0.00014  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.32755E-01 0.00013  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67626E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.82718E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.82718E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.67965E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.15819E-02 0.00178  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428818 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42891E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42891E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.65000E+00 ;
RUNNING_TIME              (idx, 1)        =  1.39598E+00 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.96667E-02  2.96667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.83883E-01  6.83883E-01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.52200E-01  2.32500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.38717E+00  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 5.48001 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.80747E+00 0.00027 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  4.91159E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  2.31036E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.81690E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.19014E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.31036E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.81690E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.22115E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.17886E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.22115E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.17886E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.33795E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.18915E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.48264E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.66895E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20806E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.10897E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.64543E+18 0.00389  1.75442E-02 0.00383 ];
U233_FISS                 (idx, [1:   4]) = [  9.21339E+19 0.00050  9.82456E-01 6.8E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  1.17456E+20 0.00046  8.10457E-01 0.00021 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37899E+19 0.00131  9.51540E-02 0.00127 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002336 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.20061E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002336 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6072678 6.07853E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3929553 3.93337E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 105 1.05228E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002336 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.42144E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36137E+20 3.0E-06  2.36137E+20 3.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37781E+19 1.3E-07  9.37781E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.44950E+20 0.00026  1.34538E+20 0.00020  1.04115E+19 0.00207 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.38728E+20 0.00016  2.28317E+20 0.00012  1.04115E+19 0.00207 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38421E+20 0.00032  2.38421E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.51783E+22 0.00024  3.39998E+22 0.00022  1.17846E+21 0.00201 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50728E+15 0.10135 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38731E+20 0.00016 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15218E+22 0.00029 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51078E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.03754E-01 0.10095 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.80724E-01 0.00957 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.25796E-04 0.01601 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.31292E+03 0.01668 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99989E-01 1.1E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  1.69736E-01 0.08316 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  1.69735E-01 0.08316 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51804E+00 2.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99668E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.90332E-01 0.00042  2.01501E-02 0.00041  6.31349E-05 0.00903 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.90346E-01 0.00016 ];
COL_KEFF                  (idx, [1:   2]) = [  9.90491E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.90346E-01 0.00016 ];
ABS_KINF                  (idx, [1:   2]) = [  9.90356E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.39928E+00 0.00017 ];
IMP_ALF                   (idx, [1:   2]) = [  7.39978E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.18068E-03 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  9.17345E-03 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.17013E-01 0.00165 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.16864E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.26944E-03 0.00559  2.62027E-04 0.01890  8.06500E-04 0.01101  5.97393E-04 0.01304  1.27624E-03 0.00874  2.75957E-04 0.01919  5.13294E-05 0.04469 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.46895E-01 0.01510  1.21548E-02 0.00614  3.23382E-02 9.5E-05  1.06193E-01 0.00063  2.97512E-01 0.00026  1.21244E+00 0.00521  3.19321E+00 0.04547 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19635E-03 0.00812  2.62881E-04 0.02839  7.88556E-04 0.01603  5.76095E-04 0.01964  1.25650E-03 0.01314  2.62437E-04 0.02765  4.98803E-05 0.06473 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.39494E-01 0.02055  1.24757E-02 3.6E-05  3.23410E-02 0.00013  1.06228E-01 0.00087  2.97422E-01 0.00038  1.23566E+00 0.00028  6.21827E+00 0.02756 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.05553E-06 0.00115  1.05511E-06 0.00115  1.19433E-06 0.01590 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.04519E-06 0.00106  1.04478E-06 0.00106  1.18237E-06 0.01586 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.12114E-03 0.00927  2.50596E-04 0.03039  7.65383E-04 0.01853  5.66868E-04 0.02192  1.21602E-03 0.01444  2.71229E-04 0.03115  5.10491E-05 0.07168 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51420E-01 0.02372  1.24756E-02 5.0E-05  3.23435E-02 0.00019  1.06050E-01 0.00104  2.97458E-01 0.00047  1.23529E+00 0.00039  6.35755E+00 0.03996 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.37725E-07 0.01304  9.37464E-07 0.01305  8.78115E-07 0.04820 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.28955E-07 0.01304  9.28697E-07 0.01304  8.70057E-07 0.04822 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91400E-03 0.03950  2.19231E-04 0.13214  7.21409E-04 0.08352  5.51127E-04 0.09085  1.14112E-03 0.06055  2.27321E-04 0.12985  5.37890E-05 0.26020 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.51210E-01 0.08497  1.24785E-02 6.7E-05  3.23289E-02 0.00045  1.06736E-01 0.00361  2.97698E-01 0.00142  1.23694E+00 0.00096  6.72076E+00 0.10769 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92603E-03 0.03903  2.14095E-04 0.13009  7.13153E-04 0.08377  5.57017E-04 0.08898  1.16584E-03 0.05906  2.22958E-04 0.12809  5.29683E-05 0.25470 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52103E-01 0.08387  1.24785E-02 6.7E-05  3.23283E-02 0.00045  1.06717E-01 0.00358  2.97696E-01 0.00141  1.23693E+00 0.00096  6.70925E+00 0.10797 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.12718E+03 0.03754 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.04915E-06 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.03887E-06 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.14646E-03 0.00570 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.00045E+03 0.00578 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.80633E-09 0.00110 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.27019E-05 0.00554  7.27097E-05 0.00555  1.59044E-06 0.25443 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  8.26284E-05 0.01651  8.27518E-05 0.01653  1.51678E-06 0.39681 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.53643E-04 0.01541  4.53519E-04 0.01543  5.00817E-04 0.24954 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72901E+01 0.01161 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.82718E+01 0.00022  4.39539E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 04:46:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04028E+00  9.93414E-01  1.00298E+00  9.98079E-01  1.00572E+00  1.01062E+00  9.96432E-01  9.97827E-01  1.00069E+00  1.01625E+00  1.01302E+00  9.84667E-01  1.02351E+00  1.00550E+00  1.01148E+00  9.93391E-01  9.11937E-01  9.94214E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.18367E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78163E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.18139E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.32748E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67876E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.83032E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.83032E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.68627E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.10600E-02 0.00184  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428785 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42872E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42872E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.19720E+02 ;
RUNNING_TIME              (idx, 1)        =  1.27136E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.41500E-02  2.44833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.19135E+01  1.12296E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.70833E-02  5.70833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.75350E-01  2.30667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.27054E+01  2.26562E+01 ];
CPU_USAGE                 (idx, 1)        = 9.41669 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97897E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.40351E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.17497E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.63400E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.85036E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88072E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32964E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26183E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46867E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.98805E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.78195E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.92566E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  6.06881E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06239E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17507E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44425E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35391E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.16565E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54596E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.44504E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38056E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.64058E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54113E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41105E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.11800E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.68974E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23373E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.09469E-04 -2.41714E+25  2.20830E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.12421E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.66701E+18 0.00377  1.77777E-02 0.00373 ];
U233_FISS                 (idx, [1:   4]) = [  9.20669E+19 0.00050  9.81865E-01 6.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  5.05953E+14 0.21510  5.39043E-06 0.21510 ];
TH232_CAPT                (idx, [1:   4]) = [  1.18885E+20 0.00047  8.03772E-01 0.00021 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37699E+19 0.00131  9.31002E-02 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  4.09435E+14 0.23979  2.76810E-06 0.23979 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85837E+16 0.03598  1.25666E-04 0.03600 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001070 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.19955E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001070 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6120534 6.12737E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3880405 3.88449E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 131 1.31048E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001070 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.41561E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36130E+20 2.9E-06  2.36130E+20 2.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37777E+19 1.2E-07  9.37777E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.47975E+20 0.00026  1.37478E+20 0.00020  1.04978E+19 0.00210 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.41753E+20 0.00016  2.31255E+20 0.00012  1.04978E+19 0.00210 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41391E+20 0.00034  2.41391E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.56288E+22 0.00025  3.44394E+22 0.00022  1.18942E+21 0.00217 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.16803E+15 0.09946 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41756E+20 0.00016 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16730E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51171E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51171E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.41688E-01 0.08009 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.87943E-01 0.00959 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.71939E-04 0.01648 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.32022E+03 0.01306 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99987E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  2.21071E-01 0.07006 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  2.21069E-01 0.07006 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51798E+00 2.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99669E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77929E-01 0.00041  1.98990E-02 0.00039  6.21421E-05 0.00935 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.77923E-01 0.00016 ];
COL_KEFF                  (idx, [1:   2]) = [  9.78285E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.77923E-01 0.00016 ];
ABS_KINF                  (idx, [1:   2]) = [  9.77935E-01 0.00016 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.39687E+00 0.00017 ];
IMP_ALF                   (idx, [1:   2]) = [  7.39669E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.20276E-03 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  9.20182E-03 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.17604E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.17744E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.30698E-03 0.00564  2.62273E-04 0.01941  8.21159E-04 0.01128  6.09822E-04 0.01364  1.29200E-03 0.00913  2.75298E-04 0.01879  4.64258E-05 0.04996 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.32115E-01 0.01533  1.22252E-02 0.00540  3.23335E-02 9.3E-05  1.06116E-01 0.00062  2.97625E-01 0.00029  1.21878E+00 0.00432  2.73111E+00 0.05056 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.15648E-03 0.00861  2.48313E-04 0.02877  7.99787E-04 0.01730  5.71950E-04 0.02013  1.22522E-03 0.01367  2.67309E-04 0.02908  4.39009E-05 0.07177 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.34986E-01 0.02225  1.24748E-02 4.0E-05  3.23331E-02 0.00012  1.06115E-01 0.00078  2.97573E-01 0.00039  1.23444E+00 0.00029  6.09421E+00 0.02939 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.06567E-06 0.00115  1.06535E-06 0.00116  1.15648E-06 0.01611 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.04203E-06 0.00108  1.04172E-06 0.00109  1.13063E-06 0.01608 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.10510E-03 0.00957  2.52627E-04 0.03143  7.78217E-04 0.01868  5.80932E-04 0.02171  1.20611E-03 0.01484  2.46868E-04 0.03218  4.03440E-05 0.08646 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.21504E-01 0.02560  1.24749E-02 5.1E-05  3.23249E-02 0.00015  1.06160E-01 0.00109  2.97573E-01 0.00051  1.23416E+00 0.00041  6.12797E+00 0.04687 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.47745E-07 0.01302  9.47466E-07 0.01302  8.76141E-07 0.05440 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.27149E-07 0.01302  9.26876E-07 0.01301  8.57503E-07 0.05454 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.77722E-03 0.03944  2.81714E-04 0.14120  7.04283E-04 0.07435  5.77012E-04 0.08634  1.00617E-03 0.06324  1.74916E-04 0.13878  3.31289E-05 0.33064 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.08089E-01 0.10268  1.24751E-02 0.00014  3.23371E-02 0.00051  1.06618E-01 0.00336  2.98357E-01 0.00165  1.23424E+00 0.00123  6.32329E+00 0.14044 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.76361E-03 0.03886  2.81199E-04 0.13954  7.17502E-04 0.07307  5.73545E-04 0.08564  9.86879E-04 0.06126  1.73175E-04 0.13953  3.13106E-05 0.32279 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.03697E-01 0.09911  1.24751E-02 0.00014  3.23356E-02 0.00050  1.06608E-01 0.00335  2.98351E-01 0.00164  1.23424E+00 0.00123  6.32329E+00 0.14044 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.97926E+03 0.03824 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.06013E-06 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.03661E-06 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.12380E-03 0.00600 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.94789E+03 0.00608 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.73708E-09 0.00103 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.24837E-05 0.00589  7.24836E-05 0.00592  1.40876E-06 0.28554 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  8.02819E-05 0.01841  8.03447E-05 0.01846  1.22563E-06 0.33483 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.94030E-04 0.01593  3.94031E-04 0.01598  4.05209E-04 0.27974 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71938E+01 0.01122 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.83032E+01 0.00022  4.39373E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 04:58:53 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03498E+00  1.00438E+00  1.00086E+00  1.00169E+00  1.01497E+00  1.01195E+00  1.01043E+00  9.97279E-01  1.00751E+00  1.00092E+00  9.89503E-01  1.01664E+00  1.00058E+00  9.94443E-01  1.00623E+00  1.01138E+00  8.91344E-01  1.00491E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.17598E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78240E-01 6.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.19219E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.33752E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66946E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.82508E+01 0.00020  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.82508E+01 0.00020  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.63214E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.09688E-02 0.00170  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428782 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42887E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42887E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.44663E+02 ;
RUNNING_TIME              (idx, 1)        =  2.53195E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.08333E-02  2.66833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.44309E+01  1.25174E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.09433E-01  5.23500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.98800E-01  2.33667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.53112E+01  1.61015E+02 ];
CPU_USAGE                 (idx, 1)        = 9.66303 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97960E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.68028E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.97452E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.71907E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.55156E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.36679E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.66237E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57495E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.52013E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.72341E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.65394E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.77838E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.01328E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.94481E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.64065E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.55626E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.42272E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.38059E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.61774E+18 ;
CS134_ACTIVITY            (idx, 1)        =  6.97088E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.22418E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.45000E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.63478E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.41515E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.66286E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.70094E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45065E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.35492E-04 -1.18240E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.12682E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.67221E+18 0.00378  1.78363E-02 0.00373 ];
U233_FISS                 (idx, [1:   4]) = [  9.19229E+19 0.00051  9.80538E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.84073E+16 0.03762  1.96326E-04 0.03764 ];
TH232_CAPT                (idx, [1:   4]) = [  1.18622E+20 0.00049  7.93278E-01 0.00023 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37483E+19 0.00131  9.19430E-02 0.00126 ];
U235_CAPT                 (idx, [1:   4]) = [  7.85791E+15 0.05657  5.25560E-05 0.05655 ];
SM149_CAPT                (idx, [1:   4]) = [  8.05789E+16 0.01684  5.38937E-04 0.01682 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002085 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.21434E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002085 1.00121E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6147518 6.15389E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3854428 3.85811E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 139 1.39090E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002085 1.00121E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 0.00000E+00 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36139E+20 3.1E-06  2.36139E+20 3.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.2E-07  9.37766E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.49417E+20 0.00025  1.38762E+20 0.00020  1.06541E+19 0.00211 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.43193E+20 0.00015  2.32539E+20 0.00012  1.06541E+19 0.00211 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.42991E+20 0.00034  2.42991E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.57385E+22 0.00025  3.45291E+22 0.00023  1.20941E+21 0.00218 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.38142E+15 0.09040 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43197E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17377E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51398E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51398E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.10886E-01 0.05571 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.12479E-01 0.01132 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.53157E-04 0.01952 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.95571E+03 0.01428 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99986E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.84461E-01 0.04675 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.84456E-01 0.04675 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51810E+00 3.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.71308E-01 0.00043  1.97644E-02 0.00042  6.25571E-05 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.72163E-01 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  9.71880E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.72163E-01 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  9.72176E-01 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.37419E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.37442E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.41406E-03 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  9.40931E-03 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.20962E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.20406E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.35158E-03 0.00570  2.82948E-04 0.01915  8.13132E-04 0.01095  6.21575E-04 0.01302  1.29418E-03 0.00925  2.88321E-04 0.01889  5.14252E-05 0.04453 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.44265E-01 0.01448  1.21902E-02 0.00578  3.23374E-02 9.5E-05  1.06079E-01 0.00061  2.97669E-01 0.00027  1.21599E+00 0.00479  3.12606E+00 0.04602 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.18829E-03 0.00794  2.63243E-04 0.02738  7.71240E-04 0.01593  5.83692E-04 0.01937  1.24888E-03 0.01314  2.74293E-04 0.02707  4.69356E-05 0.06484 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45830E-01 0.02192  1.24750E-02 3.8E-05  3.23385E-02 0.00013  1.06141E-01 0.00087  2.97592E-01 0.00038  1.23553E+00 0.00029  6.25721E+00 0.02743 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.05365E-06 0.00114  1.05339E-06 0.00114  1.14391E-06 0.01579 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.02329E-06 0.00106  1.02303E-06 0.00106  1.11094E-06 0.01578 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.15589E-03 0.00898  2.59599E-04 0.03207  7.63495E-04 0.01805  5.74756E-04 0.02172  1.23149E-03 0.01489  2.78888E-04 0.03086  4.76626E-05 0.07804 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.42618E-01 0.02392  1.24758E-02 4.6E-05  3.23327E-02 0.00016  1.06071E-01 0.00105  2.97558E-01 0.00050  1.23547E+00 0.00038  6.19021E+00 0.04276 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.39985E-07 0.01302  9.40006E-07 0.01302  8.05230E-07 0.05378 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  9.13172E-07 0.01301  9.13193E-07 0.01301  7.82074E-07 0.05370 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.73273E-03 0.04160  2.40541E-04 0.13119  6.59815E-04 0.07970  4.21220E-04 0.08970  1.12418E-03 0.06309  2.43860E-04 0.12755  4.31134E-05 0.30141 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.72055E-01 0.10089  1.24758E-02 0.00013  3.23564E-02 0.00055  1.05849E-01 0.00290  2.97042E-01 0.00143  1.23636E+00 0.00137  5.70156E+00 0.12348 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.77715E-03 0.04055  2.44436E-04 0.13164  6.75368E-04 0.07797  4.35958E-04 0.08783  1.13389E-03 0.06116  2.43210E-04 0.12584  4.42953E-05 0.28646 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.69067E-01 0.09915  1.24758E-02 0.00013  3.23571E-02 0.00056  1.05862E-01 0.00292  2.96985E-01 0.00142  1.23617E+00 0.00138  5.70156E+00 0.12348 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -2.92570E+03 0.03970 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.04980E-06 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.01956E-06 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.12945E-03 0.00555 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.98185E+03 0.00559 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.54548E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.88627E-05 0.00829  6.88581E-05 0.00829  1.04775E-06 0.32177 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.94811E-05 0.02120  6.92680E-05 0.02131  1.48847E-06 0.36113 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.68465E-04 0.01880  2.68372E-04 0.01881  2.99843E-04 0.31735 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72969E+01 0.01164 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.82508E+01 0.00020  4.38058E+01 0.00031 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 05:11:43 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04260E+00  1.00262E+00  1.00914E+00  9.95014E-01  9.96798E-01  1.01010E+00  1.00537E+00  1.01051E+00  1.00927E+00  1.00485E+00  1.00079E+00  1.00261E+00  9.94247E-01  1.01318E+00  1.01103E+00  1.00317E+00  8.89203E-01  9.99508E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.17228E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78277E-01 6.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.19903E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.34398E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66293E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.81201E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.81201E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.57822E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.03185E-02 0.00184  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428702 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42897E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42897E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.71654E+02 ;
RUNNING_TIME              (idx, 1)        =  3.81427E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.13883E-01  3.30500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.71526E+01  1.27217E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.63917E-01  5.44833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.28833E-01  2.99500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.81345E+01  1.76789E+02 ];
CPU_USAGE                 (idx, 1)        = 9.74379 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97986E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.77138E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.09150E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74363E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.03511E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.36894E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.66203E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.68960E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.54455E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.76772E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.82307E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.88649E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.01836E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.88004E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.80469E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.50539E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.47321E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.42003E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.67053E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.27044E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.43967E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.52528E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.69933E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.67990E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.78366E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.69163E+16 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28660E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.00122E-03 -2.21075E+26  2.21027E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.11205E+00 0.00064 ];
TH232_FISS                (idx, [1:   4]) = [  1.65007E+18 0.00403  1.75879E-02 0.00397 ];
U233_FISS                 (idx, [1:   4]) = [  9.18400E+19 0.00049  9.79019E-01 7.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  7.33824E+16 0.01809  7.82301E-04 0.01810 ];
TH232_CAPT                (idx, [1:   4]) = [  1.16586E+20 0.00049  7.87012E-01 0.00023 ];
U233_CAPT                 (idx, [1:   4]) = [  1.37216E+19 0.00134  9.26300E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  2.64093E+16 0.03109  1.78303E-04 0.03109 ];
SM149_CAPT                (idx, [1:   4]) = [  1.29814E+17 0.01383  8.76783E-04 0.01389 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002772 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.19719E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002772 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6124253 6.12997E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3878377 3.88186E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 142 1.41875E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002772 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.11759E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36152E+20 3.1E-06  2.36152E+20 3.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.2E-07  9.37752E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.48108E+20 0.00024  1.37618E+20 0.00018  1.04897E+19 0.00214 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.41883E+20 0.00015  2.31393E+20 0.00011  1.04897E+19 0.00214 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.41661E+20 0.00035  2.41661E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54024E+22 0.00024  3.42072E+22 0.00022  1.19515E+21 0.00221 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.43255E+15 0.09333 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41886E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16417E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51392E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51392E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.49451E-01 0.05587 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.12661E-01 0.01298 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.93241E-04 0.02274 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  4.88530E+03 0.01657 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99986E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.75774E-01 0.04788 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.75768E-01 0.04788 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51827E+00 3.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.77480E-01 0.00041  1.98860E-02 0.00040  6.35445E-05 0.00913 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.77481E-01 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  9.77283E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.77481E-01 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  9.77495E-01 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.35361E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.34941E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  9.61031E-03 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  9.64743E-03 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.21923E-01 0.00167 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.22717E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.35362E-03 0.00561  2.74642E-04 0.02068  8.27919E-04 0.01139  6.04864E-04 0.01291  1.31535E-03 0.00912  2.79805E-04 0.01906  5.10313E-05 0.04518 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43746E-01 0.01481  1.20471E-02 0.00713  3.23366E-02 0.00010  1.06220E-01 0.00064  2.97632E-01 0.00028  1.21588E+00 0.00479  3.22386E+00 0.04546 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19678E-03 0.00834  2.58331E-04 0.02999  7.74207E-04 0.01675  5.72643E-04 0.01795  1.27404E-03 0.01354  2.70743E-04 0.02736  4.68198E-05 0.06244 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44816E-01 0.02021  1.24748E-02 4.0E-05  3.23344E-02 0.00013  1.06235E-01 0.00082  2.97680E-01 0.00039  1.23542E+00 0.00033  6.27959E+00 0.02729 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  1.02934E-06 0.00114  1.02898E-06 0.00115  1.13815E-06 0.01677 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  1.00604E-06 0.00106  1.00568E-06 0.00106  1.11251E-06 0.01677 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18523E-03 0.00912  2.62060E-04 0.03104  7.70807E-04 0.01897  5.74747E-04 0.02178  1.25989E-03 0.01473  2.68284E-04 0.03045  4.94417E-05 0.07124 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.48554E-01 0.02430  1.24749E-02 5.1E-05  3.23339E-02 0.00018  1.06236E-01 0.00120  2.97849E-01 0.00053  1.23565E+00 0.00045  6.08467E+00 0.04128 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  9.17126E-07 0.01319  9.16672E-07 0.01319  9.73703E-07 0.05693 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.96758E-07 0.01317  8.96314E-07 0.01317  9.51729E-07 0.05682 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.06949E-03 0.03903  2.30626E-04 0.13483  7.78892E-04 0.07492  5.27933E-04 0.09406  1.20070E-03 0.06089  2.97197E-04 0.12425  3.41513E-05 0.27279 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.48903E-01 0.08609  1.24754E-02 0.00014  3.23555E-02 0.00053  1.06243E-01 0.00318  2.97799E-01 0.00149  1.23520E+00 0.00134  6.59154E+00 0.11747 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06691E-03 0.03846  2.32634E-04 0.13453  7.78487E-04 0.07250  5.23356E-04 0.09235  1.20414E-03 0.06013  2.93172E-04 0.12129  3.51209E-05 0.26412 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.48180E-01 0.08463  1.24754E-02 0.00014  3.23567E-02 0.00053  1.06241E-01 0.00318  2.97815E-01 0.00149  1.23522E+00 0.00129  6.59154E+00 0.11747 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.37207E+03 0.03737 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  1.02521E-06 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  1.00201E-06 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19900E-03 0.00529 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.12149E+03 0.00533 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.42601E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.53987E-05 0.01184  6.53203E-05 0.01194  2.42029E-07 0.58411 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.13629E-05 0.02755  6.13239E-05 0.02761  3.28694E-07 0.66713 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.06442E-04 0.02210  2.06829E-04 0.02216  9.95296E-05 0.57990 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73267E+01 0.01177 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.81201E+01 0.00022  4.36442E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 05:24:32 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03241E+00  9.98034E-01  1.00048E+00  1.00502E+00  9.95313E-01  9.99303E-01  1.01792E+00  1.00428E+00  1.01389E+00  9.96799E-01  9.98263E-01  1.01030E+00  9.98846E-01  1.01222E+00  1.01218E+00  9.88693E-01  9.17074E-01  9.98972E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.16783E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78322E-01 6.1E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.20943E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.35387E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65347E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.78802E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.78802E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.48825E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.95218E-02 0.00171  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428687 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42898E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42898E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.98747E+02 ;
RUNNING_TIME              (idx, 1)        =  5.09744E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.40317E-01  2.64333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.98883E+01  1.27357E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.17367E-01  5.34500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.52317E-01  2.33833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.09656E+01  1.78982E+02 ];
CPU_USAGE                 (idx, 1)        = 9.78428 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97890E+00 2.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.81746E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.13382E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75938E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.81637E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.32634E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.62898E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.77475E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56384E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.09624E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.96598E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.08168E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.00061E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.87421E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.96524E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.98938E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.51489E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.45317E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.71393E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.15759E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.86070E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.53706E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.75624E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.74300E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.77151E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.67550E+16 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57321E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.86339E-03 -4.11449E+26  2.21218E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.08919E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.61862E+18 0.00397  1.72405E-02 0.00392 ];
U233_FISS                 (idx, [1:   4]) = [  9.15491E+19 0.00051  9.75200E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.56810E+17 0.00925  2.73600E-03 0.00926 ];
TH232_CAPT                (idx, [1:   4]) = [  1.13281E+20 0.00050  7.77167E-01 0.00023 ];
U233_CAPT                 (idx, [1:   4]) = [  1.36567E+19 0.00126  9.36952E-02 0.00122 ];
U235_CAPT                 (idx, [1:   4]) = [  9.50954E+16 0.01647  6.52310E-04 0.01646 ];
SM149_CAPT                (idx, [1:   4]) = [  1.72244E+17 0.01219  1.18192E-03 0.01220 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002869 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.19237E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002869 1.00119E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6084097 6.08969E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3918644 3.92210E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 128 1.29727E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002869 1.00119E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.60187E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36170E+20 3.2E-06  2.36170E+20 3.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37717E+19 1.2E-07  9.37717E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45729E+20 0.00025  1.35410E+20 0.00019  1.03185E+19 0.00216 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39500E+20 0.00015  2.29182E+20 0.00011  1.03185E+19 0.00216 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.39357E+20 0.00034  2.39357E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.48355E+22 0.00024  3.36566E+22 0.00022  1.17888E+21 0.00223 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.11014E+15 0.09648 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.39504E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14732E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51048E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51048E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.49952E-01 0.05104 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06324E-01 0.01366 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.74956E-04 0.02334 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.00812E+03 0.01479 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99987E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.09229E-01 0.04498 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.09224E-01 0.04498 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51856E+00 3.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99682E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.87498E-01 0.00042  2.00949E-02 0.00040  6.45520E-05 0.00894 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.87279E-01 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  9.86765E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.87279E-01 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87292E-01 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.30790E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.30560E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.00599E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.00795E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.27063E-01 0.00173 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.27135E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.32362E-03 0.00567  2.77708E-04 0.01799  8.10002E-04 0.01134  6.05537E-04 0.01273  1.29822E-03 0.00885  2.79478E-04 0.01886  5.26715E-05 0.04446 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45601E-01 0.01442  1.23152E-02 0.00432  3.23284E-02 0.00011  1.06061E-01 0.00058  2.97631E-01 0.00028  1.20147E+00 0.00650  3.22332E+00 0.04494 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22933E-03 0.00816  2.62592E-04 0.02801  7.79057E-04 0.01622  5.82834E-04 0.01952  1.28257E-03 0.01280  2.69990E-04 0.02901  5.22812E-05 0.06328 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46471E-01 0.01994  1.24757E-02 3.4E-05  3.23293E-02 0.00015  1.06176E-01 0.00081  2.97609E-01 0.00038  1.23710E+00 0.00041  6.18606E+00 0.02701 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.87170E-07 0.00116  9.86748E-07 0.00117  1.12023E-06 0.01769 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.74696E-07 0.00107  9.74280E-07 0.00108  1.10571E-06 0.01756 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20353E-03 0.00907  2.76903E-04 0.02999  7.83017E-04 0.01801  5.94887E-04 0.02094  1.23743E-03 0.01455  2.60406E-04 0.03177  5.08918E-05 0.07108 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44093E-01 0.02497  1.24752E-02 5.0E-05  3.23186E-02 0.00016  1.06032E-01 0.00107  2.97482E-01 0.00048  1.23694E+00 0.00051  6.28813E+00 0.03967 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.80304E-07 0.01302  8.79899E-07 0.01302  8.46007E-07 0.04650 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.69469E-07 0.01301  8.69070E-07 0.01301  8.35240E-07 0.04654 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80851E-03 0.03973  2.26022E-04 0.11978  6.81767E-04 0.07654  4.94529E-04 0.08945  1.11844E-03 0.06025  2.41989E-04 0.13268  4.57650E-05 0.29378 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.99346E-01 0.06141  1.24753E-02 0.00013  3.23129E-02 0.00038  1.06247E-01 0.00316  2.97311E-01 0.00144  1.23603E+00 0.00101  4.55059E+00 0.12823 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.83743E-03 0.03918  2.27316E-04 0.12046  6.99423E-04 0.07422  4.98327E-04 0.08665  1.12336E-03 0.05932  2.42794E-04 0.12897  4.62031E-05 0.28842 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.98284E-01 0.05901  1.24753E-02 0.00013  3.23125E-02 0.00038  1.06277E-01 0.00319  2.97332E-01 0.00143  1.23598E+00 0.00101  4.55059E+00 0.12823 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.23060E+03 0.03813 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.82669E-07 0.00062 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.70268E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.15153E-03 0.00593 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.20834E+03 0.00598 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.32169E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.36295E-05 0.01306  6.36002E-05 0.01313  5.85680E-07 0.39581 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.91274E-05 0.02974  5.89783E-05 0.02985  6.76436E-07 0.43287 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84846E-04 0.02302  1.84756E-04 0.02310  2.30745E-04 0.37818 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74182E+01 0.01162 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.78802E+01 0.00021  4.32294E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 05:37:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02753E+00  1.00692E+00  1.01733E+00  1.00471E+00  1.00288E+00  9.98068E-01  1.01025E+00  1.01123E+00  9.91070E-01  1.01084E+00  1.00869E+00  1.00961E+00  1.00907E+00  9.89743E-01  9.99051E-01  9.97221E-01  8.99581E-01  1.00621E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.16778E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78322E-01 6.2E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.21632E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.36061E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64970E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.76883E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.76883E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.42171E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.93145E-02 0.00170  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428780 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42884E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42884E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.24979E+02 ;
RUNNING_TIME              (idx, 1)        =  6.37232E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.69250E-01  2.89333E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.25378E+01  1.26494E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.70633E-01  5.32667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.81283E-01  2.88667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.37116E+01  1.79073E+02 ];
CPU_USAGE                 (idx, 1)        = 9.80770 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97978E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.84386E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.07011E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74904E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.42495E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.26669E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.58415E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.77103E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55835E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.17944E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.01790E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.35945E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.76490E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.41804E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.04107E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.43247E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.48921E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.43442E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.68707E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.79339E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.24448E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.47558E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.73754E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.81055E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.63445E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.66046E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85981E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.76421E-03 -6.10354E+26  2.21417E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.07293E+00 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.59290E+18 0.00411  1.69762E-02 0.00408 ];
U233_FISS                 (idx, [1:   4]) = [  9.10852E+19 0.00049  9.70752E-01 8.8E-05 ];
U235_FISS                 (idx, [1:   4]) = [  5.07233E+17 0.00698  5.40516E-03 0.00694 ];
TH232_CAPT                (idx, [1:   4]) = [  1.10641E+20 0.00047  7.70168E-01 0.00023 ];
U233_CAPT                 (idx, [1:   4]) = [  1.35561E+19 0.00129  9.43654E-02 0.00125 ];
U235_CAPT                 (idx, [1:   4]) = [  1.84448E+17 0.01120  1.28405E-03 0.01120 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82425E+17 0.01135  1.26991E-03 0.01135 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001896 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.19747E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001896 1.00120E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 6049917 6.05620E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3951817 3.95562E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 162 1.61708E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001896 1.00120E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.45286E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36175E+20 3.2E-06  2.36175E+20 3.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37674E+19 1.1E-07  9.37674E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43656E+20 0.00024  1.33476E+20 0.00018  1.01797E+19 0.00217 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37423E+20 0.00015  2.27243E+20 0.00010  1.01797E+19 0.00217 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.37208E+20 0.00033  2.37208E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43576E+22 0.00024  3.31914E+22 0.00021  1.16627E+21 0.00221 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.83971E+15 0.09053 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.37427E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.13246E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50662E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50662E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.19708E-01 0.05271 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13722E-01 0.01354 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.73336E-04 0.02463 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.03054E+03 0.01525 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 1.5E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.94660E-01 0.04674 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.94654E-01 0.04674 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51873E+00 3.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99691E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.96257E-01 0.00041  2.02658E-02 0.00039  6.67699E-05 0.00890 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.95930E-01 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  9.95718E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.95930E-01 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95947E-01 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.27530E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.27367E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.03927E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.04062E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.30162E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.30514E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.32919E-03 0.00565  2.63639E-04 0.01978  8.12631E-04 0.01114  6.01456E-04 0.01307  1.31407E-03 0.00886  2.82638E-04 0.01821  5.47637E-05 0.04169 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.60799E-01 0.01504  1.21905E-02 0.00578  3.23236E-02 0.00011  1.06135E-01 0.00061  2.97678E-01 0.00028  1.21332E+00 0.00541  3.67914E+00 0.04176 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24145E-03 0.00784  2.52006E-04 0.02914  7.93261E-04 0.01595  5.83317E-04 0.01935  1.28472E-03 0.01239  2.74337E-04 0.02776  5.38034E-05 0.05673 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.62001E-01 0.02062  1.24754E-02 3.8E-05  3.23256E-02 0.00015  1.06126E-01 0.00079  2.97619E-01 0.00037  1.23820E+00 0.00040  6.70406E+00 0.02436 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.57411E-07 0.00107  9.57084E-07 0.00107  1.06119E-06 0.01777 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.53706E-07 0.00097  9.53381E-07 0.00097  1.05709E-06 0.01773 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.28100E-03 0.00902  2.56974E-04 0.03375  7.97199E-04 0.01742  6.10964E-04 0.02007  1.29473E-03 0.01458  2.73764E-04 0.03120  4.73740E-05 0.07325 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44160E-01 0.02595  1.24759E-02 4.5E-05  3.23192E-02 0.00016  1.06280E-01 0.00109  2.97782E-01 0.00048  1.23838E+00 0.00060  6.12882E+00 0.04189 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.49040E-07 0.01297  8.48655E-07 0.01297  8.24825E-07 0.04480 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.46031E-07 0.01296  8.45646E-07 0.01296  8.22253E-07 0.04484 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89521E-03 0.03878  2.22316E-04 0.13364  7.24420E-04 0.07689  5.08899E-04 0.08260  1.14315E-03 0.05890  2.50726E-04 0.11937  4.57046E-05 0.26536 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.55556E-01 0.07303  1.24750E-02 0.00015  3.23000E-02 0.00050  1.05616E-01 0.00228  2.98380E-01 0.00153  1.23941E+00 0.00157  5.63256E+00 0.12198 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.91234E-03 0.03801  2.15271E-04 0.13397  7.37629E-04 0.07456  5.16103E-04 0.08104  1.15273E-03 0.05795  2.44765E-04 0.11689  4.58475E-05 0.28493 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52535E-01 0.07281  1.24750E-02 0.00015  3.23000E-02 0.00049  1.05635E-01 0.00231  2.98418E-01 0.00154  1.23945E+00 0.00157  5.63256E+00 0.12198 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.43659E+03 0.03705 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.51009E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.47334E-07 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29633E-03 0.00577 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.46728E+03 0.00583 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.27038E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.34483E-05 0.01370  6.33428E-05 0.01379  9.18992E-07 0.34740 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.91190E-05 0.02786  5.91562E-05 0.02790  8.20645E-07 0.53229 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85838E-04 0.02343  1.85558E-04 0.02350  2.89648E-04 0.33817 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68433E+01 0.01143 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.76883E+01 0.00021  4.29197E+01 0.00031 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 05:50:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04142E+00  1.00287E+00  1.01003E+00  9.98989E-01  1.00632E+00  9.98646E-01  1.00646E+00  1.00776E+00  9.98669E-01  1.01680E+00  9.98029E-01  1.01025E+00  1.00865E+00  1.00212E+00  1.00691E+00  9.92518E-01  8.87944E-01  1.00561E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.17043E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78296E-01 6.3E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.22258E-01 0.00015  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.36692E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64543E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.73563E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.73563E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.32968E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.85901E-02 0.00182  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428686 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42902E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42902E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.52328E+02 ;
RUNNING_TIME              (idx, 1)        =  7.65868E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.94117E-01  2.48667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.52976E+01  1.27599E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.24633E-01  5.40000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.03850E-01  2.24833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.65787E+01  1.78405E+02 ];
CPU_USAGE                 (idx, 1)        = 9.82321 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98079E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.86158E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.07552E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.76063E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.09335E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.20882E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.53920E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.83460E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57475E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.35197E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.15720E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  6.28004E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.55531E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  7.18938E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.20067E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.22706E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.52177E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.45955E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.72027E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.60928E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.92954E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.46585E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.80933E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.78293E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.56184E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.63806E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43302E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -4.44116E-03 -9.80635E+26  2.21787E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.04815E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.54692E+18 0.00393  1.64977E-02 0.00389 ];
U233_FISS                 (idx, [1:   4]) = [  9.01173E+19 0.00050  9.61137E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  1.10183E+18 0.00474  1.17531E-02 0.00476 ];
PU239_FISS                (idx, [1:   4]) = [  9.46907E+13 0.49896  1.00416E-06 0.49894 ];
TH232_CAPT                (idx, [1:   4]) = [  1.06563E+20 0.00049  7.58344E-01 0.00025 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34090E+19 0.00132  9.54234E-02 0.00125 ];
U235_CAPT                 (idx, [1:   4]) = [  3.99207E+17 0.00761  2.84104E-03 0.00761 ];
U238_CAPT                 (idx, [1:   4]) = [  2.32886E+13 1.00000  1.64810E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.91717E+17 0.01066  1.36441E-03 0.01066 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003144 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.18550E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003144 1.00119E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5999478 6.00493E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4003512 4.00677E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 154 1.54016E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003144 1.00119E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.11759E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36163E+20 3.3E-06  2.36163E+20 3.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37580E+19 1.1E-07  9.37580E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40542E+20 0.00025  1.30641E+20 0.00019  9.90028E+18 0.00221 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34300E+20 0.00015  2.24399E+20 0.00011  9.90028E+18 0.00221 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.34009E+20 0.00033  2.34009E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36648E+22 0.00024  3.25242E+22 0.00021  1.14066E+21 0.00214 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.60795E+15 0.10156 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.34303E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10942E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49791E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49791E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.72156E-01 0.05545 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24636E-01 0.01291 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.72327E-04 0.02409 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.01123E+03 0.01655 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99985E-01 1.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.82014E-01 0.04846 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.82007E-01 0.04846 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51886E+00 3.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99711E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.00902E+00 0.00041  2.05318E-02 0.00040  6.56792E-05 0.00840 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.00914E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.00928E+00 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.00914E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00915E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.22647E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.22708E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.09129E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.09025E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.36569E-01 0.00164 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.36080E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.26313E-03 0.00556  2.62387E-04 0.01901  7.95189E-04 0.01119  5.93842E-04 0.01243  1.27898E-03 0.00867  2.77113E-04 0.01915  5.56220E-05 0.04173 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.63296E-01 0.01517  1.21906E-02 0.00578  3.23172E-02 0.00010  1.06162E-01 0.00066  2.97937E-01 0.00028  1.22005E+00 0.00502  3.61356E+00 0.04139 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20707E-03 0.00798  2.52912E-04 0.02882  7.86332E-04 0.01611  5.86808E-04 0.01915  1.25632E-03 0.01275  2.67198E-04 0.02808  5.75042E-05 0.06291 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.58553E-01 0.02024  1.24754E-02 4.0E-05  3.23148E-02 0.00014  1.06125E-01 0.00084  2.97842E-01 0.00038  1.24130E+00 0.00058  6.47542E+00 0.02482 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.16836E-07 0.00109  9.16588E-07 0.00109  9.89828E-07 0.01618 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  9.24990E-07 0.00100  9.24740E-07 0.00100  9.98665E-07 0.01618 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18743E-03 0.00854  2.44752E-04 0.03292  7.68276E-04 0.01790  5.97251E-04 0.02028  1.24637E-03 0.01373  2.76338E-04 0.03184  5.44465E-05 0.07016 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.61959E-01 0.02439  1.24761E-02 4.7E-05  3.23204E-02 0.00019  1.06116E-01 0.00101  2.97954E-01 0.00049  1.24069E+00 0.00072  6.55697E+00 0.03658 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.13996E-07 0.01300  8.13886E-07 0.01301  7.42184E-07 0.04797 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  8.21758E-07 0.01300  8.21648E-07 0.01300  7.49215E-07 0.04802 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93217E-03 0.03872  2.37240E-04 0.12758  7.13375E-04 0.07577  5.56208E-04 0.08000  1.12592E-03 0.05787  2.46515E-04 0.12411  5.29198E-05 0.23656 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.55686E-01 0.08409  1.24765E-02 0.00013  3.23263E-02 0.00047  1.05849E-01 0.00246  2.97710E-01 0.00140  1.24542E+00 0.00246  5.78596E+00 0.11741 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93557E-03 0.03777  2.31346E-04 0.12604  7.18065E-04 0.07461  5.67399E-04 0.07737  1.12020E-03 0.05734  2.45683E-04 0.12291  5.28709E-05 0.24248 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.57264E-01 0.08450  1.24765E-02 0.00013  3.23279E-02 0.00048  1.05859E-01 0.00248  2.97705E-01 0.00139  1.24542E+00 0.00246  5.78596E+00 0.11741 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.63131E+03 0.03670 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.11006E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  9.19112E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18691E-03 0.00549 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.49935E+03 0.00553 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.19356E-09 0.00070 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.40256E-05 0.01329  6.39550E-05 0.01339  5.40847E-07 0.42631 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.80624E-05 0.02936  5.80719E-05 0.02939  4.81470E-07 0.61540 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.83521E-04 0.02302  1.83521E-04 0.02310  2.06894E-04 0.41358 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71516E+01 0.01185 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.73563E+01 0.00022  4.24275E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 06:02:55 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02189E+00  1.00962E+00  1.00831E+00  1.00753E+00  1.01021E+00  1.00323E+00  1.00857E+00  1.00839E+00  1.00274E+00  9.94514E-01  1.00563E+00  1.00190E+00  9.98334E-01  9.93622E-01  1.00465E+00  9.97854E-01  9.02983E-01  1.02003E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.17613E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78239E-01 6.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.23063E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37518E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64234E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.69093E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.69093E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.20753E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.80376E-02 0.00181  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428275 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42878E+04 0.00052 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42878E+04 0.00052 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.78740E+02 ;
RUNNING_TIME              (idx, 1)        =  8.93570E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.20400E-01  2.62833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  8.79653E+01  1.26677E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.78733E-01  5.41000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.26800E-01  2.28500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.93485E+01  1.79230E+02 ];
CPU_USAGE                 (idx, 1)        = 9.83403 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98113E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.87429E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  7.06293E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.77298E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.73615E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.11803E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.46798E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.91314E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.59460E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.84068E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.43577E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.65605E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.29030E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.58866E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.50354E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.36746E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.54383E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.47429E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.74078E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.41200E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.31533E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.43438E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.93435E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.86792E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.42235E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61315E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28660E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -8.26233E-03 -1.82437E+27  2.22631E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02080E+00 0.00064 ];
TH232_FISS                (idx, [1:   4]) = [  1.48969E+18 0.00392  1.59004E-02 0.00385 ];
U233_FISS                 (idx, [1:   4]) = [  8.78251E+19 0.00050  9.37504E-01 0.00012 ];
U235_FISS                 (idx, [1:   4]) = [  2.68459E+18 0.00274  2.86570E-02 0.00269 ];
PU239_FISS                (idx, [1:   4]) = [  1.93503E+15 0.10925  2.06586E-05 0.10934 ];
PU240_FISS                (idx, [1:   4]) = [  2.29507E+13 1.00000  2.46221E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01070E+20 0.00049  7.37563E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.30281E+19 0.00128  9.50756E-02 0.00124 ];
U235_CAPT                 (idx, [1:   4]) = [  9.69341E+17 0.00488  7.07387E-03 0.00487 ];
U238_CAPT                 (idx, [1:   4]) = [  2.41210E+13 1.00000  1.75199E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  7.13283E+14 0.18176  5.21741E-06 0.18163 ];
PU240_CAPT                (idx, [1:   4]) = [  1.14065E+14 0.44596  8.38938E-07 0.44597 ];
SM149_CAPT                (idx, [1:   4]) = [  1.87697E+17 0.01079  1.36969E-03 0.01078 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001426 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15410E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001426 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5940131 5.94629E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4061137 4.06509E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 158 1.58020E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001426 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.23517E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36094E+20 3.6E-06  2.36094E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37340E+19 1.0E-07  9.37340E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37045E+20 0.00024  1.27420E+20 0.00018  9.62549E+18 0.00214 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30779E+20 0.00014  2.21154E+20 0.00010  9.62549E+18 0.00214 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30451E+20 0.00033  2.30451E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.28705E+22 0.00024  3.17528E+22 0.00021  1.11765E+21 0.00220 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.64660E+15 0.08896 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30783E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08219E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47443E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47443E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.21219E-01 0.05329 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.28626E-01 0.01242 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.72120E-04 0.02398 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.08816E+03 0.01643 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.03646E-01 0.04688 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.03640E-01 0.04688 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51877E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99762E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02377E+00 0.00039  2.08291E-02 0.00038  6.74817E-05 0.00917 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02421E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02457E+00 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02421E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02423E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.15570E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.15676E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.17136E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.16967E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47423E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.46701E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.27672E-03 0.00570  2.55959E-04 0.02017  7.94919E-04 0.01130  5.85980E-04 0.01291  1.29771E-03 0.00878  2.83331E-04 0.01834  5.88130E-05 0.04225 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.75073E-01 0.01513  1.22444E-02 0.00520  3.22870E-02 0.00011  1.06403E-01 0.00059  2.98766E-01 0.00030  1.22635E+00 0.00482  3.82952E+00 0.03961 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28433E-03 0.00782  2.55145E-04 0.02894  8.00258E-04 0.01609  5.90329E-04 0.01813  1.29598E-03 0.01206  2.83000E-04 0.02668  5.96151E-05 0.06109 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.74856E-01 0.02290  1.24761E-02 3.7E-05  3.22877E-02 0.00014  1.06444E-01 0.00081  2.98803E-01 0.00042  1.24593E+00 0.00068  6.73019E+00 0.02263 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.68665E-07 0.00109  8.68317E-07 0.00109  9.82401E-07 0.01782 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.89218E-07 0.00102  8.88863E-07 0.00102  1.00561E-06 0.01778 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22575E-03 0.00926  2.57859E-04 0.03281  7.86691E-04 0.01846  5.71913E-04 0.02175  1.28070E-03 0.01378  2.75466E-04 0.03003  5.31176E-05 0.06884 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.67201E-01 0.02488  1.24764E-02 4.5E-05  3.22873E-02 0.00020  1.06487E-01 0.00116  2.99029E-01 0.00052  1.24655E+00 0.00096  6.73576E+00 0.03386 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.71083E-07 0.01299  7.70860E-07 0.01299  7.23371E-07 0.04562 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.89449E-07 0.01298  7.89220E-07 0.01298  7.40708E-07 0.04565 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.85404E-03 0.04015  2.35518E-04 0.13050  6.63896E-04 0.07793  4.34733E-04 0.08519  1.20277E-03 0.06154  2.70206E-04 0.11874  4.69164E-05 0.27311 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.72050E-01 0.08979  1.24747E-02 0.00016  3.22674E-02 0.00054  1.06613E-01 0.00314  2.98318E-01 0.00143  1.24678E+00 0.00252  7.14133E+00 0.09145 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87161E-03 0.03906  2.36439E-04 0.12841  6.62100E-04 0.07517  4.39752E-04 0.08313  1.21124E-03 0.06033  2.73025E-04 0.11674  4.90572E-05 0.25712 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.77988E-01 0.08926  1.24747E-02 0.00016  3.22658E-02 0.00053  1.06600E-01 0.00313  2.98390E-01 0.00143  1.24693E+00 0.00251  7.14133E+00 0.09145 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.69963E+03 0.03781 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.63203E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.83623E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25963E-03 0.00617 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.77682E+03 0.00615 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.09845E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.29646E-05 0.01360  6.28463E-05 0.01369  4.88660E-07 0.45552 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.89372E-05 0.02778  5.86961E-05 0.02781  7.24416E-07 0.48649 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.82126E-04 0.02321  1.82225E-04 0.02326  1.69210E-04 0.44656 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67814E+01 0.01153 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.69093E+01 0.00022  4.17041E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 06:15:43 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03040E+00  1.00370E+00  9.98505E-01  1.01021E+00  1.00504E+00  1.00537E+00  1.00035E+00  1.00796E+00  1.01724E+00  1.01216E+00  1.01528E+00  9.99328E-01  9.96160E-01  1.01547E+00  9.91803E-01  1.00819E+00  8.90846E-01  9.91997E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.18400E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78160E-01 6.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.23417E-01 0.00015  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37914E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64349E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.65319E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.65319E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.11726E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.81176E-02 0.00184  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428980 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42884E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42884E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.00538E+03 ;
RUNNING_TIME              (idx, 1)        =  1.02156E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.52167E-01  3.17667E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.00651E+02  1.26861E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.32317E-01  5.35833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.54033E-01  2.71333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.02148E+02  1.78666E+02 ];
CPU_USAGE                 (idx, 1)        = 9.84156 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98154E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88291E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 181 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.96237E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.75795E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.16904E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01464E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38891E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.91626E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.58781E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.29657E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.85012E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.19744E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.18096E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.06728E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.92575E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.43276E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.47138E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.41399E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.66132E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.67182E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.35432E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.32974E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99743E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.11116E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.18121E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.60389E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57638E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.68179E-02 -3.71349E+27  2.24520E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.00435E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.42812E+18 0.00398  1.52372E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  8.46420E+19 0.00050  9.03073E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  5.01984E+18 0.00211  5.35588E-02 0.00206 ];
PU239_FISS                (idx, [1:   4]) = [  3.03076E+16 0.02739  3.23309E-04 0.02739 ];
PU240_FISS                (idx, [1:   4]) = [  5.72969E+14 0.19657  6.08499E-06 0.19657 ];
PU241_FISS                (idx, [1:   4]) = [  9.83834E+14 0.16485  1.04844E-05 0.16509 ];
TH232_CAPT                (idx, [1:   4]) = [  9.64484E+19 0.00049  7.10994E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.25428E+19 0.00135  9.24624E-02 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  1.80409E+18 0.00353  1.32998E-02 0.00352 ];
U238_CAPT                 (idx, [1:   4]) = [  2.75549E+14 0.28642  2.03544E-06 0.28642 ];
PU239_CAPT                (idx, [1:   4]) = [  1.47657E+16 0.04081  1.08910E-04 0.04082 ];
PU240_CAPT                (idx, [1:   4]) = [  2.70352E+15 0.08901  1.99079E-05 0.08899 ];
PU241_CAPT                (idx, [1:   4]) = [  1.58784E+14 0.37644  1.17412E-06 0.37643 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82650E+17 0.01141  1.34642E-03 0.01140 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001850 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12792E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001850 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5914830 5.92047E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4086848 4.09064E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 172 1.71799E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001850 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.47035E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35983E+20 3.6E-06  2.35983E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36954E+19 9.2E-08  9.36954E+19 9.2E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35628E+20 0.00024  1.26091E+20 0.00018  9.53643E+18 0.00215 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29323E+20 0.00014  2.19787E+20 0.00010  9.53643E+18 0.00215 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.29127E+20 0.00033  2.29127E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24756E+22 0.00023  3.13642E+22 0.00021  1.11135E+21 0.00215 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.94198E+15 0.09494 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.29327E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06730E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.43697E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.43697E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.71596E-01 0.05470 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.32330E-01 0.01268 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.80203E-04 0.02486 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.18840E+03 0.01534 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99983E-01 1.6E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.01850E-01 0.04731 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.01844E-01 0.04731 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51862E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99845E+02 9.2E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02997E+00 0.00040  2.09580E-02 0.00039  6.83059E-05 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.03019E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.03000E+00 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.03019E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03021E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.08434E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.08303E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.25793E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.25918E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.59983E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.59936E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33056E-03 0.00554  2.49932E-04 0.01912  7.85846E-04 0.01103  6.06928E-04 0.01248  1.32656E-03 0.00883  3.01003E-04 0.01800  6.02920E-05 0.03942 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.88589E-01 0.01440  1.20842E-02 0.00681  3.22560E-02 0.00012  1.06387E-01 0.00056  2.99894E-01 0.00030  1.24590E+00 0.00330  4.21727E+00 0.03672 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.31618E-03 0.00768  2.48358E-04 0.02762  7.77942E-04 0.01552  5.98811E-04 0.01812  1.33061E-03 0.01257  3.03703E-04 0.02517  5.67524E-05 0.05412 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.86634E-01 0.02028  1.24764E-02 4.4E-05  3.22541E-02 0.00016  1.06354E-01 0.00074  2.99799E-01 0.00041  1.25479E+00 0.00095  7.17233E+00 0.01935 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.29203E-07 0.00107  8.28846E-07 0.00107  9.35030E-07 0.01491 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.53977E-07 0.00103  8.53608E-07 0.00103  9.63092E-07 0.01494 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24077E-03 0.00910  2.42059E-04 0.03259  7.77101E-04 0.01836  5.84646E-04 0.02121  1.27196E-03 0.01436  3.02250E-04 0.02879  6.27513E-05 0.06280 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.96977E-01 0.02441  1.24761E-02 9.6E-05  3.22493E-02 0.00023  1.06226E-01 0.00093  3.00106E-01 0.00056  1.25380E+00 0.00113  7.06004E+00 0.02878 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.36028E-07 0.01303  7.35595E-07 0.01303  7.69466E-07 0.04550 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.58361E-07 0.01302  7.57917E-07 0.01302  7.92741E-07 0.04552 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.05836E-03 0.03786  2.57143E-04 0.12375  7.26105E-04 0.07719  5.69400E-04 0.08142  1.14332E-03 0.06038  2.95739E-04 0.10861  6.66473E-05 0.21847 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.55071E-01 0.09936  1.24737E-02 0.00016  3.22275E-02 0.00046  1.05903E-01 0.00222  3.00217E-01 0.00164  1.25510E+00 0.00324  7.75926E+00 0.06510 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.07208E-03 0.03735  2.58737E-04 0.12099  7.26395E-04 0.07678  5.72371E-04 0.07955  1.14975E-03 0.05986  3.01006E-04 0.10670  6.38248E-05 0.21648 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.52854E-01 0.09856  1.24736E-02 0.00016  3.22266E-02 0.00045  1.05912E-01 0.00224  3.00338E-01 0.00164  1.25493E+00 0.00323  7.75913E+00 0.06510 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.18533E+03 0.03641 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.24757E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.49373E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.27365E-03 0.00599 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.96885E+03 0.00592 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.02400E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.37645E-05 0.01265  6.37271E-05 0.01275  3.20661E-07 0.53148 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.98556E-05 0.02961  5.97999E-05 0.02967  3.97066E-07 0.56224 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.91685E-04 0.02409  1.91924E-04 0.02415  1.25898E-04 0.50151 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64665E+01 0.01128 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.65319E+01 0.00021  4.09778E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 06:28:42 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03562E+00  1.00868E+00  1.00561E+00  1.00132E+00  9.85573E-01  9.99203E-01  1.00543E+00  9.99935E-01  1.01472E+00  1.00180E+00  1.01068E+00  1.00388E+00  1.00904E+00  1.00068E+00  9.99397E-01  1.00760E+00  9.04054E-01  1.00678E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.18564E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78144E-01 6.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.23497E-01 0.00015  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.38002E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64401E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.64105E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.64104E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.08989E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.82747E-02 0.00176  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428834 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42886E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42886E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.13384E+03 ;
RUNNING_TIME              (idx, 1)        =  1.15138E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.82583E-01  3.04167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.13518E+02  1.28663E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  4.86717E-01  5.44000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.80450E-01  2.63333E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.15130E+02  1.78984E+02 ];
CPU_USAGE                 (idx, 1)        = 9.84758 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98195E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.88956E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.86758E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73477E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.92143E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97031E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35744E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86600E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56798E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.30374E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.99626E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.12293E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.21343E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.14529E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.06791E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.86442E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.39807E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35697E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.58399E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.67595E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.78022E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24799E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.94953E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.45540E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.03318E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.60107E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21968E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.18390E-02 -4.82218E+27  2.25628E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.98985E-01 0.00065 ];
TH232_FISS                (idx, [1:   4]) = [  1.40734E+18 0.00420  1.50239E-02 0.00416 ];
U233_FISS                 (idx, [1:   4]) = [  8.35535E+19 0.00050  8.92016E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  5.71373E+18 0.00205  6.09994E-02 0.00200 ];
PU239_FISS                (idx, [1:   4]) = [  6.36856E+16 0.01932  6.79351E-04 0.01925 ];
PU240_FISS                (idx, [1:   4]) = [  2.31147E+15 0.09839  2.46713E-05 0.09833 ];
PU241_FISS                (idx, [1:   4]) = [  2.92092E+15 0.09134  3.11681E-05 0.09131 ];
TH232_CAPT                (idx, [1:   4]) = [  9.49381E+19 0.00050  7.01620E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23746E+19 0.00136  9.14522E-02 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  2.05907E+18 0.00323  1.52173E-02 0.00321 ];
U238_CAPT                 (idx, [1:   4]) = [  2.93212E+14 0.27499  2.17543E-06 0.27499 ];
PU239_CAPT                (idx, [1:   4]) = [  2.97100E+16 0.02925  2.19653E-04 0.02924 ];
PU240_CAPT                (idx, [1:   4]) = [  6.90563E+15 0.05871  5.10063E-05 0.05871 ];
PU241_CAPT                (idx, [1:   4]) = [  6.39948E+14 0.19194  4.72476E-06 0.19207 ];
SM149_CAPT                (idx, [1:   4]) = [  1.78936E+17 0.01137  1.32249E-03 0.01138 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002052 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13899E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002052 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5910297 5.91596E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4091604 4.09527E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 151 1.51268E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002052 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.68575E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35976E+20 3.7E-06  2.35976E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36811E+19 8.9E-08  9.36811E+19 8.9E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35319E+20 0.00024  1.25779E+20 0.00018  9.53941E+18 0.00209 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29000E+20 0.00014  2.19461E+20 0.00011  9.53941E+18 0.00209 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.28725E+20 0.00032  2.28725E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.23538E+22 0.00023  3.12407E+22 0.00021  1.11305E+21 0.00218 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.45923E+15 0.09110 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.29003E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06266E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42510E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42510E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.26286E-01 0.06052 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.34920E-01 0.01191 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.68073E-04 0.02424 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.19150E+03 0.01599 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99985E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.46173E-01 0.05321 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.46168E-01 0.05321 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51892E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99875E+02 8.9E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.03141E+00 0.00040  2.09817E-02 0.00039  7.08701E-05 0.00844 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.03162E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.03177E+00 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.03162E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03163E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.05998E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.06162E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.28889E-02 0.00124 ];
IMP_EALF                  (idx, [1:   2]) = [  1.28645E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.64305E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.64154E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.40238E-03 0.00521  2.52155E-04 0.01919  8.11478E-04 0.01147  6.13488E-04 0.01229  1.34807E-03 0.00836  3.09378E-04 0.01781  6.78049E-05 0.03757 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.02448E-01 0.01438  1.21726E-02 0.00597  3.22477E-02 0.00012  1.06447E-01 0.00056  3.00203E-01 0.00031  1.24749E+00 0.00388  4.60671E+00 0.03347 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.38488E-03 0.00768  2.42657E-04 0.02709  7.97612E-04 0.01680  6.25491E-04 0.01742  1.34402E-03 0.01168  3.08705E-04 0.02493  6.63935E-05 0.05468 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.03520E-01 0.02064  1.24753E-02 4.1E-05  3.22441E-02 0.00015  1.06352E-01 0.00069  3.00026E-01 0.00040  1.25995E+00 0.00092  7.20006E+00 0.01867 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.18167E-07 0.00111  8.17830E-07 0.00112  9.21366E-07 0.01529 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.43764E-07 0.00103  8.43415E-07 0.00103  9.50549E-07 0.01535 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.36850E-03 0.00850  2.49224E-04 0.03201  8.07429E-04 0.01779  6.16217E-04 0.02034  1.32571E-03 0.01328  3.05398E-04 0.02997  6.45230E-05 0.06362 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.94469E-01 0.02384  1.24754E-02 5.2E-05  3.22496E-02 0.00019  1.06466E-01 0.00093  3.00236E-01 0.00051  1.25842E+00 0.00133  7.20707E+00 0.02694 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.22202E-07 0.01296  7.22045E-07 0.01296  7.02933E-07 0.04872 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.45128E-07 0.01296  7.44967E-07 0.01296  7.25163E-07 0.04864 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98722E-03 0.03720  2.25994E-04 0.11853  6.51668E-04 0.07787  5.62352E-04 0.08719  1.21905E-03 0.05571  2.65756E-04 0.10835  6.23962E-05 0.27923 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.05677E-01 0.07830  1.24764E-02 0.00013  3.22530E-02 0.00057  1.07105E-01 0.00308  3.00789E-01 0.00166  1.25739E+00 0.00313  6.28826E+00 0.09246 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.01506E-03 0.03636  2.34581E-04 0.11365  6.50678E-04 0.07553  5.62635E-04 0.08631  1.23056E-03 0.05421  2.73473E-04 0.10730  6.31269E-05 0.28005 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.03962E-01 0.07735  1.24763E-02 0.00013  3.22531E-02 0.00057  1.07107E-01 0.00307  3.00818E-01 0.00165  1.25735E+00 0.00313  6.29520E+00 0.09231 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.15656E+03 0.03526 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.13674E-07 0.00061 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.39135E-07 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37386E-03 0.00496 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.14654E+03 0.00493 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.98770E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.29271E-05 0.01374  6.28243E-05 0.01382  7.26074E-07 0.36302 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.84288E-05 0.03059  5.83037E-05 0.03069  7.59991E-07 0.44614 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.80151E-04 0.02359  1.79970E-04 0.02364  2.47685E-04 0.35554 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.66221E+01 0.01126 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.64104E+01 0.00022  4.07426E+01 0.00032 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 06:42:00 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03390E+00  1.01229E+00  9.83244E-01  1.00368E+00  1.01587E+00  9.96590E-01  1.01554E+00  1.00154E+00  9.98145E-01  9.98740E-01  1.01229E+00  1.00018E+00  1.00639E+00  9.91672E-01  1.00979E+00  1.00840E+00  9.08462E-01  1.00327E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.18981E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78102E-01 6.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.23236E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37778E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64838E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.62784E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.62784E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.07309E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.74868E-02 0.00182  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428617 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42892E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42892E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.26546E+03 ;
RUNNING_TIME              (idx, 1)        =  1.28436E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.10183E-01  2.76000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.26701E+02  1.31837E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.38533E-01  5.18167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.03433E-01  2.29000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.28428E+02  1.80211E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85277 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98219E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89510E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 185 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.85545E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73255E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.00578E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.95119E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34415E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87304E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56715E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.46270E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.17001E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.20473E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.37746E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.22030E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.22484E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.24427E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38317E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34441E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56753E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.68521E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.15791E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23130E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96288E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.75493E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99611E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.60057E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86299E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.69843E-02 -5.95829E+27  2.26764E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.96838E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.40101E+18 0.00408  1.49516E-02 0.00402 ];
U233_FISS                 (idx, [1:   4]) = [  8.27635E+19 0.00053  8.83317E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  6.20642E+18 0.00196  6.62401E-02 0.00190 ];
U238_FISS                 (idx, [1:   4]) = [  4.56623E+13 0.70669  4.95245E-07 0.70675 ];
PU239_FISS                (idx, [1:   4]) = [  1.04548E+17 0.01407  1.11597E-03 0.01407 ];
PU240_FISS                (idx, [1:   4]) = [  4.94167E+15 0.06776  5.27269E-05 0.06769 ];
PU241_FISS                (idx, [1:   4]) = [  7.09265E+15 0.05766  7.57638E-05 0.05767 ];
TH232_CAPT                (idx, [1:   4]) = [  9.39375E+19 0.00050  6.94768E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22430E+19 0.00137  9.05545E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  2.22537E+18 0.00319  1.64597E-02 0.00318 ];
U238_CAPT                 (idx, [1:   4]) = [  3.41916E+14 0.25562  2.53599E-06 0.25561 ];
PU239_CAPT                (idx, [1:   4]) = [  5.24496E+16 0.02060  3.87996E-04 0.02061 ];
PU240_CAPT                (idx, [1:   4]) = [  1.47833E+16 0.04028  1.09423E-04 0.04029 ];
PU241_CAPT                (idx, [1:   4]) = [  1.64285E+15 0.11341  1.21565E-05 0.11341 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80551E+17 0.01136  1.33535E-03 0.01135 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002414 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10894E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002414 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5907809 5.91318E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4094465 4.09777E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 140 1.39919E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002414 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 8.19564E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.35991E+20 3.9E-06  2.35991E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36682E+19 9.0E-08  9.36682E+19 9.0E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.35170E+20 0.00024  1.25785E+20 0.00018  9.38482E+18 0.00221 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.28838E+20 0.00014  2.19454E+20 0.00010  9.38482E+18 0.00221 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.28653E+20 0.00032  2.28653E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.23027E+22 0.00022  3.12067E+22 0.00020  1.09595E+21 0.00223 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.20405E+15 0.08809 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.28842E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05926E+22 0.00029 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.41476E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.41476E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.46104E-01 0.05682 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.42215E-01 0.01200 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.71429E-04 0.02492 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.17852E+03 0.01597 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99986E-01 1.2E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.78882E-01 0.04966 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.78877E-01 0.04966 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.51944E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99903E+02 9.0E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.03236E+00 0.00041  2.09991E-02 0.00040  7.06071E-05 0.00862 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.03241E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.03217E+00 0.00031 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.03241E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03242E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04228E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04140E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31199E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31268E-02 0.00085 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.68295E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.68070E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39432E-03 0.00562  2.68457E-04 0.01902  7.98935E-04 0.01120  6.14859E-04 0.01246  1.34941E-03 0.00863  3.01421E-04 0.01848  6.12385E-05 0.04139 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.85974E-01 0.01523  1.22810E-02 0.00478  3.22349E-02 0.00013  1.06505E-01 0.00052  3.00575E-01 0.00031  1.24219E+00 0.00462  4.11052E+00 0.03715 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35011E-03 0.00785  2.63462E-04 0.02739  7.77604E-04 0.01642  6.23239E-04 0.01785  1.33596E-03 0.01239  2.90582E-04 0.02582  5.92614E-05 0.05602 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.79191E-01 0.01992  1.24776E-02 4.5E-05  3.22283E-02 0.00017  1.06459E-01 0.00068  3.00650E-01 0.00042  1.25990E+00 0.00095  6.89687E+00 0.02031 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.09620E-07 0.00108  8.09309E-07 0.00108  9.03823E-07 0.01444 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.35717E-07 0.00100  8.35396E-07 0.00100  9.32788E-07 0.01442 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.33941E-03 0.00881  2.69854E-04 0.03027  7.80348E-04 0.01795  5.96296E-04 0.02016  1.33836E-03 0.01412  2.96247E-04 0.02976  5.83021E-05 0.06834 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.86438E-01 0.02587  1.24768E-02 4.7E-05  3.22397E-02 0.00021  1.06618E-01 0.00099  3.00349E-01 0.00053  1.25933E+00 0.00122  7.35518E+00 0.02800 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.15329E-07 0.01297  7.15059E-07 0.01297  7.17138E-07 0.04696 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.38838E-07 0.01297  7.38560E-07 0.01297  7.40660E-07 0.04696 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01098E-03 0.03730  2.62679E-04 0.12485  6.63948E-04 0.07455  4.91402E-04 0.08599  1.29828E-03 0.05783  2.49613E-04 0.11859  4.50550E-05 0.23548 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.88875E-01 0.08433  1.24769E-02 0.00012  3.22393E-02 0.00070  1.06633E-01 0.00275  3.00636E-01 0.00163  1.25745E+00 0.00435  7.83800E+00 0.06946 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99741E-03 0.03674  2.58183E-04 0.12116  6.64936E-04 0.07489  4.87450E-04 0.08441  1.29400E-03 0.05637  2.48811E-04 0.11721  4.40222E-05 0.23534 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.84028E-01 0.08178  1.24768E-02 0.00012  3.22355E-02 0.00070  1.06648E-01 0.00276  3.00619E-01 0.00162  1.25749E+00 0.00435  7.83800E+00 0.06946 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.21282E+03 0.03492 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.04653E-07 0.00062 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.30591E-07 0.00045 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32348E-03 0.00501 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.13107E+03 0.00503 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.96660E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.39521E-05 0.01368  6.39498E-05 0.01368  5.30457E-07 0.47588 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.85825E-05 0.02911  5.85260E-05 0.02918  4.72740E-07 0.50877 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.82612E-04 0.02428  1.82731E-04 0.02429  1.54170E-04 0.44798 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65239E+01 0.01064 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.62784E+01 0.00023  4.05677E+01 0.00033 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 06:55:38 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03545E+00  1.01364E+00  1.00988E+00  9.99887E-01  1.00378E+00  1.00030E+00  9.97668E-01  9.93083E-01  9.99407E-01  1.00240E+00  1.00586E+00  1.00578E+00  1.00173E+00  1.02016E+00  1.00922E+00  1.00326E+00  8.92231E-01  1.00627E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.19540E-02 0.00029  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.78046E-01 6.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.23108E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.37688E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65322E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.61451E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.61450E+01 0.00023  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.05064E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.79498E-02 0.00180  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429066 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42892E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42892E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.39976E+03 ;
RUNNING_TIME              (idx, 1)        =  1.42063E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.40533E-01  3.03500E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.40154E+02  1.34526E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  6.48583E-01  1.10050E-01 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.28000E-01  2.44833E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.42055E+02  1.82027E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85310 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98230E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.89981E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.87788E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73748E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.89962E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92278E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32820E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92388E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57367E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.02720E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  5.74715E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.78386E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.02697E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44699E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.71206E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.36692E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37337E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33404E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55516E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.72034E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.29670E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22875E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01349E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.37848E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98165E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.60603E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43619E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -4.75918E-02 -1.05086E+28  2.31315E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.95966E-01 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.37763E+18 0.00414  1.47071E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  8.07302E+19 0.00052  8.61887E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  7.16544E+18 0.00175  7.64985E-02 0.00168 ];
U238_FISS                 (idx, [1:   4]) = [  4.61295E+13 0.70665  4.94915E-07 0.70666 ];
PU239_FISS                (idx, [1:   4]) = [  3.39521E+17 0.00864  3.62493E-03 0.00865 ];
PU240_FISS                (idx, [1:   4]) = [  2.45100E+16 0.03149  2.61642E-04 0.03145 ];
PU241_FISS                (idx, [1:   4]) = [  4.28694E+16 0.02257  4.57534E-04 0.02252 ];
TH232_CAPT                (idx, [1:   4]) = [  9.19254E+19 0.00051  6.75851E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.19128E+19 0.00132  8.75875E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  2.57940E+18 0.00293  1.89644E-02 0.00291 ];
U238_CAPT                 (idx, [1:   4]) = [  1.21954E+15 0.13481  8.96336E-06 0.13487 ];
PU239_CAPT                (idx, [1:   4]) = [  1.66896E+17 0.01148  1.22704E-03 0.01148 ];
PU240_CAPT                (idx, [1:   4]) = [  7.39021E+16 0.01740  5.43369E-04 0.01740 ];
PU241_CAPT                (idx, [1:   4]) = [  9.13529E+15 0.04946  6.71970E-05 0.04944 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81973E+17 0.01027  1.33809E-03 0.01027 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002408 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09687E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002408 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5923057 5.92828E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4079215 4.08255E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 136 1.35801E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002408 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.30385E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36151E+20 4.2E-06  2.36151E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36274E+19 8.8E-08  9.36274E+19 8.8E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36103E+20 0.00025  1.26618E+20 0.00018  9.48565E+18 0.00222 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.29731E+20 0.00015  2.20245E+20 0.00010  9.48565E+18 0.00222 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.29433E+20 0.00032  2.29433E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.23860E+22 0.00023  3.12769E+22 0.00020  1.10902E+21 0.00225 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.11656E+15 0.09266 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.29734E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.05980E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.37518E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.37518E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.08247E-01 0.06059 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.34486E-01 0.01231 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.68786E-04 0.02448 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.23274E+03 0.01598 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99986E-01 1.3E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.47375E-01 0.05304 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.47370E-01 0.05304 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52225E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99990E+02 8.8E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02975E+00 0.00040  2.09438E-02 0.00038  7.13077E-05 0.00825 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02908E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02936E+00 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02908E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02910E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00056E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.99945E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.36792E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36894E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.77207E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.77148E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44800E-03 0.00558  2.48412E-04 0.01875  8.13986E-04 0.01115  6.13472E-04 0.01281  1.39355E-03 0.00868  3.13032E-04 0.01751  6.55498E-05 0.03784 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.98504E-01 0.01409  1.21934E-02 0.00579  3.22199E-02 0.00015  1.06616E-01 0.00057  3.00982E-01 0.00029  1.24567E+00 0.00442  4.44913E+00 0.03408 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.40455E-03 0.00760  2.39065E-04 0.02797  7.92886E-04 0.01466  6.12997E-04 0.01867  1.38240E-03 0.01219  3.15816E-04 0.02528  6.13833E-05 0.05234 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.92913E-01 0.01844  1.24788E-02 8.3E-05  3.22150E-02 0.00019  1.06682E-01 0.00075  3.00867E-01 0.00040  1.26090E+00 0.00114  7.04287E+00 0.01862 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  7.95096E-07 0.00116  7.94835E-07 0.00116  8.77314E-07 0.01372 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.18641E-07 0.00106  8.18372E-07 0.00107  9.03362E-07 0.01372 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39702E-03 0.00842  2.46536E-04 0.03153  8.00635E-04 0.01820  5.95925E-04 0.01924  1.38306E-03 0.01352  3.07019E-04 0.02888  6.38447E-05 0.06349 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.89506E-01 0.02315  1.24782E-02 8.5E-05  3.22216E-02 0.00024  1.06665E-01 0.00099  3.01195E-01 0.00052  1.25994E+00 0.00157  6.78029E+00 0.03048 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.02848E-07 0.01302  7.02570E-07 0.01302  6.94117E-07 0.04393 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23912E-07 0.01301  7.23626E-07 0.01301  7.14884E-07 0.04390 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.93944E-03 0.03905  2.42800E-04 0.12194  7.59946E-04 0.07241  5.35001E-04 0.09064  1.08872E-03 0.05775  2.51219E-04 0.12331  6.17490E-05 0.26451 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.57604E-01 0.07578  1.24769E-02 0.00013  3.22131E-02 0.00072  1.06815E-01 0.00304  3.01824E-01 0.00176  1.26342E+00 0.00381  6.74277E+00 0.08476 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94001E-03 0.03841  2.42051E-04 0.12085  7.63113E-04 0.07109  5.35369E-04 0.08923  1.09462E-03 0.05712  2.45718E-04 0.11983  5.91340E-05 0.26167 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52011E-01 0.07395  1.24769E-02 0.00013  3.22122E-02 0.00072  1.06812E-01 0.00303  3.01792E-01 0.00175  1.26327E+00 0.00385  6.74277E+00 0.08476 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.22422E+03 0.03704 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  7.89506E-07 0.00063 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12904E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.33915E-03 0.00530 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.23150E+03 0.00539 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93422E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.35010E-05 0.01338  6.34976E-05 0.01340  6.80672E-07 0.37727 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.61467E-05 0.02909  5.62273E-05 0.02912  5.81536E-07 0.46131 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.79458E-04 0.02380  1.79274E-04 0.02394  2.54851E-04 0.41274 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63746E+01 0.01153 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.61450E+01 0.00023  4.01711E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 07:09:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02565E+00  1.00166E+00  9.98764E-01  9.99278E-01  1.01385E+00  1.01531E+00  9.98455E-01  1.00340E+00  1.00876E+00  1.00918E+00  9.99164E-01  9.98775E-01  1.01097E+00  9.95630E-01  1.01086E+00  1.00743E+00  8.92912E-01  1.00996E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.20728E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77927E-01 6.8E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.21818E-01 0.00017  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.36502E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66596E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.60331E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.60331E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.07672E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  5.89733E-02 0.00172  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1429230 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42899E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42899E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.53461E+03 ;
RUNNING_TIME              (idx, 1)        =  1.55722E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.70417E-01  2.98833E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.53692E+02  1.35380E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.04000E-01  5.54167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.51283E-01  2.32000E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.55714E+02  1.82790E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85479 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97896E+00 7.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90362E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.87794E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73229E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.25721E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89751E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31909E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.94920E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56940E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.61885E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38548E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.44511E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14170E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69206E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23483E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58087E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34405E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31048E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52372E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.74884E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.60289E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20698E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.98976E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.14082E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.95175E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61161E+16 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28692E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.01721E-01 -2.24606E+28  2.43267E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.96801E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.35771E+18 0.00413  1.45026E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  7.93057E+19 0.00053  8.47276E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  7.44085E+18 0.00185  7.94950E-02 0.00178 ];
U238_FISS                 (idx, [1:   4]) = [  1.61592E+14 0.37642  1.72710E-06 0.37644 ];
PU239_FISS                (idx, [1:   4]) = [  6.74847E+17 0.00579  7.20978E-03 0.00577 ];
PU240_FISS                (idx, [1:   4]) = [  7.90509E+16 0.01796  8.44465E-04 0.01794 ];
PU241_FISS                (idx, [1:   4]) = [  1.50683E+17 0.01218  1.61037E-03 0.01222 ];
TH232_CAPT                (idx, [1:   4]) = [  9.05708E+19 0.00051  6.61696E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.16999E+19 0.00141  8.54782E-02 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  2.65897E+18 0.00286  1.94275E-02 0.00287 ];
U238_CAPT                 (idx, [1:   4]) = [  2.67396E+15 0.09390  1.95676E-05 0.09392 ];
PU239_CAPT                (idx, [1:   4]) = [  3.27762E+17 0.00836  2.39471E-03 0.00837 ];
PU240_CAPT                (idx, [1:   4]) = [  2.25125E+17 0.01008  1.64495E-03 0.01008 ];
PU241_CAPT                (idx, [1:   4]) = [  3.24165E+16 0.02575  2.36930E-04 0.02578 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81514E+17 0.01132  1.32636E-03 0.01133 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002926 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09244E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002926 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5940226 5.94522E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4062550 4.06556E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 150 1.49564E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002926 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.37836E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36466E+20 4.4E-06  2.36466E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35850E+19 9.4E-08  9.35850E+19 9.4E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36864E+20 0.00024  1.27284E+20 0.00018  9.58025E+18 0.00219 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30449E+20 0.00014  2.20869E+20 0.00010  9.58025E+18 0.00219 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30230E+20 0.00032  2.30230E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.26155E+22 0.00022  3.14934E+22 0.00020  1.12208E+21 0.00222 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.44536E+15 0.09239 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30453E+20 0.00014 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06091E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29521E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29521E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.30926E-01 0.05724 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.32218E-01 0.01234 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.77015E-04 0.02401 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.13631E+03 0.01674 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99985E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.71619E-01 0.05028 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.71614E-01 0.05028 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52675E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00080E+02 9.4E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02710E+00 0.00040  2.08952E-02 0.00039  6.96932E-05 0.00841 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02722E+00 0.00014 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02716E+00 0.00031 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02722E+00 0.00014 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02724E+00 0.00014 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.97154E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.97206E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.40819E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.40692E-02 0.00083 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.84210E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83547E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.42064E-03 0.00542  2.44498E-04 0.02065  7.79766E-04 0.01159  6.15510E-04 0.01291  1.39472E-03 0.00831  3.21951E-04 0.01776  6.41921E-05 0.04028 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05778E-01 0.01513  1.21787E-02 0.00597  3.21818E-02 0.00016  1.06677E-01 0.00057  3.01149E-01 0.00030  1.24701E+00 0.00396  4.31761E+00 0.03565 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.38908E-03 0.00719  2.45396E-04 0.02820  7.84219E-04 0.01586  6.11869E-04 0.01796  1.36525E-03 0.01149  3.22166E-04 0.02492  6.01872E-05 0.05712 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.01730E-01 0.02071  1.24819E-02 0.00014  3.21836E-02 0.00021  1.06673E-01 0.00074  3.01091E-01 0.00042  1.26074E+00 0.00125  7.14829E+00 0.01903 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  7.92642E-07 0.00111  7.92250E-07 0.00112  9.13177E-07 0.01616 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14025E-07 0.00103  8.13622E-07 0.00103  9.37864E-07 0.01615 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.33351E-03 0.00873  2.28083E-04 0.03286  7.54690E-04 0.01838  6.02994E-04 0.01963  1.36987E-03 0.01329  3.15771E-04 0.02808  6.20983E-05 0.06387 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.09983E-01 0.02438  1.24794E-02 0.00012  3.21876E-02 0.00030  1.06664E-01 0.00094  3.01067E-01 0.00052  1.25875E+00 0.00172  7.34890E+00 0.02655 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.03200E-07 0.01298  7.03083E-07 0.01298  6.92616E-07 0.04239 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22586E-07 0.01297  7.22465E-07 0.01298  7.11849E-07 0.04238 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.02683E-03 0.03705  1.58291E-04 0.13518  6.76898E-04 0.07683  5.89764E-04 0.08076  1.22967E-03 0.05536  3.15865E-04 0.11306  5.63397E-05 0.21917 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.45129E-01 0.09349  1.24793E-02 7.4E-05  3.22020E-02 0.00078  1.06641E-01 0.00245  3.00761E-01 0.00162  1.26365E+00 0.00424  8.25702E+00 0.05812 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03654E-03 0.03617  1.61616E-04 0.13384  6.82316E-04 0.07381  5.93910E-04 0.07952  1.23129E-03 0.05433  3.13250E-04 0.11106  5.41551E-05 0.21197 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.34726E-01 0.09233  1.24793E-02 7.4E-05  3.22028E-02 0.00078  1.06642E-01 0.00245  3.00782E-01 0.00162  1.26390E+00 0.00425  8.25820E+00 0.05813 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.31523E+03 0.03447 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  7.88151E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09420E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40775E-03 0.00490 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.32450E+03 0.00491 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93124E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.45581E-05 0.01326  6.46175E-05 0.01325  2.86929E-07 0.50150 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.76985E-05 0.03015  5.77191E-05 0.03017  4.17478E-07 0.58079 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.89314E-04 0.02324  1.89458E-04 0.02322  1.57839E-04 0.45072 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.57168E+01 0.01167 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.60331E+01 0.00022  3.98651E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 07:23:01 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02993E+00  1.00207E+00  1.00943E+00  9.93278E-01  1.00604E+00  1.00450E+00  1.01093E+00  1.00962E+00  1.00399E+00  9.89756E-01  1.01399E+00  9.99843E-01  1.00295E+00  1.00640E+00  1.00755E+00  1.01038E+00  8.95197E-01  1.00414E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.21532E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77847E-01 6.7E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.20649E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.35408E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.68137E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.60855E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.60855E+01 0.00021  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.13175E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.02446E-02 0.00174  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428929 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42899E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42899E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.67031E+03 ;
RUNNING_TIME              (idx, 1)        =  1.69451E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.98317E-01  2.79000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.67284E+02  1.35925E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.72367E-01  6.83667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.74617E-01  2.32500E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.69442E+02  1.83077E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85721 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98231E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90665E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.88047E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73074E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.71021E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89282E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31796E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95631E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56786E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.72389E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55939E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.54281E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16279E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76244E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.38709E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.92299E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33601E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30443E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51538E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.76139E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00608E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20131E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97869E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34382E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94673E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61312E+16 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93022E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.57152E-01 -3.47002E+28  2.55506E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.95067E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.36061E+18 0.00427  1.45321E-02 0.00420 ];
U233_FISS                 (idx, [1:   4]) = [  7.92183E+19 0.00052  8.46208E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.38769E+18 0.00189  7.89097E-02 0.00178 ];
U238_FISS                 (idx, [1:   4]) = [  1.82826E+14 0.35183  1.94118E-06 0.35183 ];
PU239_FISS                (idx, [1:   4]) = [  7.18703E+17 0.00574  7.67792E-03 0.00574 ];
PU240_FISS                (idx, [1:   4]) = [  9.13103E+16 0.01546  9.75295E-04 0.01545 ];
PU241_FISS                (idx, [1:   4]) = [  1.77855E+17 0.01117  1.89973E-03 0.01116 ];
TH232_CAPT                (idx, [1:   4]) = [  9.02932E+19 0.00053  6.58728E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17089E+19 0.00133  8.54237E-02 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  2.64728E+18 0.00296  1.93122E-02 0.00290 ];
U238_CAPT                 (idx, [1:   4]) = [  3.31172E+15 0.08117  2.41898E-05 0.08115 ];
PU239_CAPT                (idx, [1:   4]) = [  3.55186E+17 0.00805  2.59158E-03 0.00806 ];
PU240_CAPT                (idx, [1:   4]) = [  2.63443E+17 0.00921  1.92214E-03 0.00922 ];
PU241_CAPT                (idx, [1:   4]) = [  4.10850E+16 0.02398  2.99824E-04 0.02399 ];
SM149_CAPT                (idx, [1:   4]) = [  1.80377E+17 0.01101  1.31604E-03 0.01102 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002946 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07360E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002946 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5943323 5.94812E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4059447 4.06244E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 176 1.76014E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002946 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.91155E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36530E+20 4.6E-06  2.36530E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35781E+19 9.7E-08  9.35781E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37115E+20 0.00024  1.27338E+20 0.00018  9.77727E+18 0.00217 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30693E+20 0.00015  2.20916E+20 0.00010  9.77727E+18 0.00217 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30446E+20 0.00033  2.30446E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.28463E+22 0.00023  3.17073E+22 0.00021  1.13897E+21 0.00211 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  4.06452E+15 0.08325 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30697E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06309E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.22495E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.22495E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.96505E-01 0.05887 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.32782E-01 0.01224 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.82063E-04 0.02314 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.22953E+03 0.01541 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99982E-01 1.5E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.66699E-01 0.05075 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.66693E-01 0.05075 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52762E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00095E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02670E+00 0.00042  2.08869E-02 0.00041  6.91111E-05 0.00891 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02640E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02648E+00 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02640E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02642E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.97169E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.97155E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.40796E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.40767E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.84774E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83942E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.41674E-03 0.00551  2.42186E-04 0.01979  8.20177E-04 0.01124  5.94269E-04 0.01257  1.36918E-03 0.00826  3.20667E-04 0.01698  7.02605E-05 0.03912 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13338E-01 0.01500  1.21052E-02 0.00665  3.21944E-02 0.00016  1.06525E-01 0.00053  3.01296E-01 0.00030  1.25010E+00 0.00321  4.47833E+00 0.03425 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35368E-03 0.00753  2.36794E-04 0.02826  7.93512E-04 0.01553  5.85839E-04 0.01872  1.36104E-03 0.01138  3.04284E-04 0.02456  7.22069E-05 0.05377 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.20797E-01 0.02143  1.24793E-02 0.00011  3.21891E-02 0.00022  1.06549E-01 0.00074  3.01355E-01 0.00041  1.25774E+00 0.00163  7.10989E+00 0.01870 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  7.95201E-07 0.00111  7.94884E-07 0.00111  8.89941E-07 0.01429 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16331E-07 0.00102  8.16006E-07 0.00103  9.13532E-07 0.01427 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30677E-03 0.00906  2.30045E-04 0.03326  7.79275E-04 0.01773  5.79878E-04 0.02024  1.33995E-03 0.01368  3.11126E-04 0.02844  6.64963E-05 0.06439 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.23454E-01 0.02586  1.24821E-02 0.00019  3.21876E-02 0.00027  1.06466E-01 0.00090  3.01309E-01 0.00052  1.26142E+00 0.00184  7.58672E+00 0.02427 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.02089E-07 0.01294  7.01716E-07 0.01294  7.08188E-07 0.04902 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20945E-07 0.01293  7.20561E-07 0.01293  7.27435E-07 0.04904 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83116E-03 0.03909  2.13234E-04 0.14480  6.80223E-04 0.07723  5.04133E-04 0.08449  1.09878E-03 0.06232  2.74025E-04 0.11291  6.07653E-05 0.23446 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.17777E-01 0.08713  1.24871E-02 0.00081  3.21616E-02 0.00096  1.06220E-01 0.00241  2.99670E-01 0.00154  1.26367E+00 0.00491  7.09019E+00 0.07381 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.83281E-03 0.03827  2.07897E-04 0.14155  6.68670E-04 0.07493  5.04067E-04 0.08344  1.11102E-03 0.06094  2.78723E-04 0.11049  6.24396E-05 0.22187 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.22418E-01 0.08561  1.24871E-02 0.00081  3.21617E-02 0.00096  1.06228E-01 0.00242  2.99741E-01 0.00154  1.26360E+00 0.00494  7.09019E+00 0.07381 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.06367E+03 0.03753 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  7.90415E-07 0.00062 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11422E-07 0.00046 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30225E-03 0.00627 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.17969E+03 0.00634 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.95120E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.43252E-05 0.01258  6.42080E-05 0.01268  5.84791E-07 0.43274 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.81736E-05 0.02777  5.81844E-05 0.02782  2.98114E-07 0.48974 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.94664E-04 0.02228  1.94731E-04 0.02226  1.84324E-04 0.40910 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60012E+01 0.01095 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.60855E+01 0.00021  3.98540E+01 0.00034 ];


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
INPUT_FILE_NAME           (idx, [1: 28])  = 'msfr_mix1_benchmark_burn_12c' ;
WORKING_DIRECTORY         (idx, [1: 33])  = '/gv/data/cfx/tiago/doutorado/msfr' ;
HOSTNAME                  (idx, [1: 26])  = 'playerone.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549403.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb  8 04:33:34 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Feb  8 07:33:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1644305614541 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04195E+00  1.01798E+00  1.01941E+00  1.00867E+00  9.97964E-01  1.00392E+00  1.00247E+00  9.99439E-01  9.97838E-01  1.01101E+00  1.00471E+00  1.00209E+00  9.96626E-01  1.01136E+00  1.01325E+00  1.00776E+00  8.67602E-01  9.95951E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.22345E-02 0.00028  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.77766E-01 6.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.19181E-01 0.00016  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.34024E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69683E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.61115E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.61115E+01 0.00022  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  9.19382E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  6.11903E-02 0.00172  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 1428945 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.42893E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.42893E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.77839E+03 ;
RUNNING_TIME              (idx, 1)        =  1.80400E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  6.82350E-01  6.82350E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.29233E-01  3.09167E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.78110E+02  1.08252E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  8.23800E-01  5.14333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.00967E-01  2.62667E-02 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.80392E+02  1.80392E+02 ];
CPU_USAGE                 (idx, 1)        = 9.85805 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97925E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.90758E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 257886.47 ;
ALLOC_MEMSIZE             (idx, 1)        = 7808.42;
MEMSIZE                   (idx, 1)        = 7646.57;
XS_MEMSIZE                (idx, 1)        = 7342.01;
MAT_MEMSIZE               (idx, 1)        = 174.83;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 117.63;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 161.84;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 8 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 301761 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.88231E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73018E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.98316E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.89288E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31857E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95810E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56726E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.73855E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60791E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.55528E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16693E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78267E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43115E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.01965E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33334E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30234E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51257E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.75548E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.12946E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.20018E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.97447E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.38791E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.94753E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.61386E+16 0.00031  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57353E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.13222E-01 -4.70807E+28  2.67887E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.94830E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.34335E+18 0.00416  1.43495E-02 0.00411 ];
U233_FISS                 (idx, [1:   4]) = [  7.91913E+19 0.00054  8.45982E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.40601E+18 0.00181  7.91165E-02 0.00174 ];
U238_FISS                 (idx, [1:   4]) = [  1.58580E+14 0.37641  1.69639E-06 0.37636 ];
PU239_FISS                (idx, [1:   4]) = [  7.27393E+17 0.00551  7.77126E-03 0.00551 ];
PU240_FISS                (idx, [1:   4]) = [  9.28675E+16 0.01566  9.91941E-04 0.01565 ];
PU241_FISS                (idx, [1:   4]) = [  1.85594E+17 0.01075  1.98318E-03 0.01076 ];
TH232_CAPT                (idx, [1:   4]) = [  9.02454E+19 0.00051  6.57841E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17070E+19 0.00144  8.53371E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  2.64027E+18 0.00293  1.92472E-02 0.00292 ];
U238_CAPT                 (idx, [1:   4]) = [  3.11710E+15 0.08348  2.27066E-05 0.08352 ];
PU239_CAPT                (idx, [1:   4]) = [  3.54844E+17 0.00808  2.58664E-03 0.00806 ];
PU240_CAPT                (idx, [1:   4]) = [  2.69812E+17 0.00944  1.96719E-03 0.00945 ];
PU241_CAPT                (idx, [1:   4]) = [  3.89600E+16 0.02325  2.84019E-04 0.02326 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84364E+17 0.01056  1.34411E-03 0.01057 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002530 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06463E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002530 1.00106E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5945296 5.95026E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 4057075 4.06023E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 159 1.59978E+02 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002530 1.00106E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.30385E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52494E-02 1.5E-09  3.52494E-02 1.5E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36537E+20 4.2E-06  2.36537E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35770E+19 1.0E-07  9.35770E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37216E+20 0.00025  1.27364E+20 0.00018  9.85220E+18 0.00211 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30793E+20 0.00015  2.20941E+20 0.00010  9.85220E+18 0.00211 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.30552E+20 0.00031  2.30552E+20 0.00031  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.30637E+22 0.00022  3.19164E+22 0.00019  1.14731E+21 0.00209 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.69212E+15 0.08937 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.30797E+20 0.00015 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06416E+22 0.00029 ];
INI_FMASS                 (idx, 1)        =  8.51078E+04 ;
TOT_FMASS                 (idx, 1)        =  8.15591E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51078E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.15591E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.16816E-01 0.05873 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.37319E-01 0.01239 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.81117E-04 0.02425 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.25330E+03 0.01494 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99984E-01 1.4E-06 ];
SIX_FF_LT                 (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
SIX_FF_KINF               (idx, [1:   2]) = [  3.69543E-01 0.05044 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.69537E-01 0.05044 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52773E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00098E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.02617E+00 0.00042  2.08750E-02 0.00040  6.96620E-05 0.00874 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.02599E+00 0.00015 ];
COL_KEFF                  (idx, [1:   2]) = [  1.02603E+00 0.00031 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.02599E+00 0.00015 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02600E+00 0.00015 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.97523E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.97546E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.40300E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.40213E-02 0.00082 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83156E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83562E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44797E-03 0.00540  2.51906E-04 0.02023  8.00507E-04 0.01136  6.23938E-04 0.01273  1.38897E-03 0.00861  3.13915E-04 0.01828  6.87288E-05 0.03737 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.07273E-01 0.01449  1.21236E-02 0.00649  3.21960E-02 0.00016  1.06558E-01 0.00153  3.01356E-01 0.00030  1.24448E+00 0.00401  4.59292E+00 0.03352 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.39804E-03 0.00724  2.44821E-04 0.02735  7.77989E-04 0.01529  6.09076E-04 0.01770  1.38376E-03 0.01169  3.11388E-04 0.02563  7.10047E-05 0.05115 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.11307E-01 0.01963  1.24795E-02 0.00012  3.21989E-02 0.00022  1.06726E-01 0.00072  3.01267E-01 0.00041  1.25668E+00 0.00156  7.18900E+00 0.01837 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.02181E-07 0.00113  8.01731E-07 0.00114  9.43837E-07 0.01631 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.23070E-07 0.00105  8.22608E-07 0.00105  9.68290E-07 0.01627 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.31747E-03 0.00876  2.39165E-04 0.03227  7.67562E-04 0.01787  5.94763E-04 0.02036  1.34446E-03 0.01384  3.07369E-04 0.03021  6.41507E-05 0.06438 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04424E-01 0.02423  1.24796E-02 0.00015  3.21922E-02 0.00029  1.06611E-01 0.00098  3.01235E-01 0.00052  1.25675E+00 0.00207  7.38299E+00 0.02590 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.05301E-07 0.01303  7.04995E-07 0.01303  7.08722E-07 0.04355 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23971E-07 0.01301  7.23658E-07 0.01302  7.27346E-07 0.04356 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.88594E-03 0.04002  1.97660E-04 0.13901  6.70985E-04 0.07812  5.21603E-04 0.08968  1.21987E-03 0.05824  2.40639E-04 0.12644  3.51828E-05 0.26137 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.63278E-01 0.09267  1.24750E-02 0.00017  3.22198E-02 0.00088  1.06790E-01 0.00292  3.01067E-01 0.00163  1.27073E+00 0.00518  6.49720E+00 0.11267 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.88876E-03 0.03906  1.94697E-04 0.13924  6.73771E-04 0.07555  5.24732E-04 0.08739  1.21515E-03 0.05727  2.45326E-04 0.12394  3.50875E-05 0.25898 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.68028E-01 0.09271  1.24750E-02 0.00017  3.22205E-02 0.00088  1.06785E-01 0.00292  3.01126E-01 0.00162  1.27066E+00 0.00519  6.49720E+00 0.11267 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.11431E+03 0.03812 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  7.96701E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.17448E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.28965E-03 0.00589 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.12941E+03 0.00587 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.97113E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.58825E-05 0.01220  6.57989E-05 0.01229  5.84162E-07 0.42548 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.82629E-05 0.02647  5.83742E-05 0.02654  3.33697E-07 0.56514 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.94506E-04 0.02297  1.94477E-04 0.02293  2.02277E-04 0.37728 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61609E+01 0.01104 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.61115E+01 0.00022  3.98578E+01 0.00034 ];

