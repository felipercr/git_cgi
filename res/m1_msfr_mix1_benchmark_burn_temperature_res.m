
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 08:15:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01731E+00  1.00172E+00  9.99788E-01  9.96256E-01  1.00243E+00  9.91157E-01  9.96246E-01  1.00848E+00  9.97834E-01  9.96859E-01  9.97393E-01  1.00476E+00  9.95463E-01  9.92837E-01  9.89907E-01  9.96521E-01  1.01043E+00  1.00462E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.12503E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58750E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40137E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53967E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66129E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45378E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44472E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.11223E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.74486E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334104 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33389E+04 0.00056 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33389E+04 0.00056 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.24819E+02 ;
RUNNING_TIME              (idx, 1)        =  3.40822E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.71183E-01  1.71183E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.27749E+01  3.27749E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.93750E-01  6.39083E-01 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.34805E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.53045 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98660E+00 3.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.49766E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

NORM_COEF                 (idx, [1:   4]) = [  7.15513E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.83930E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.54233E+18 0.00398  1.64272E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  9.23469E+19 0.00049  9.83573E-01 6.6E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.35020E+19 0.00050  7.58673E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34303E+19 0.00133  1.08975E-01 0.00130 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001656 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16035E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001656 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5162182 5.16741E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3932675 3.93661E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906799 9.07576E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001656 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.16650E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36451E+20 3.7E-06  2.36451E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37768E+19 1.2E-07  9.37768E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23262E+20 0.00030  1.10045E+20 0.00022  1.32175E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17039E+20 0.00017  2.03822E+20 0.00012  1.32175E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38504E+20 0.00037  2.38504E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.14532E+22 0.00025  2.98431E+22 0.00023  1.61008E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.16469E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38686E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06124E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.25627E-01 0.08544 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.88742E-01 0.00955 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.94513E-04 0.01606 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.14983E+04 0.02468 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09249E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.07572E-01 0.07493 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.70618E-01 0.07493 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52142E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.92486E-01 0.00040  1.09932E-01 0.00038  3.53634E-04 0.00911 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.91790E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.91429E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.91790E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09083E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01963E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01843E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34157E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34303E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.47921E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48561E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.24464E-03 0.00555  2.66367E-04 0.01930  8.09211E-04 0.01059  5.90250E-04 0.01298  1.26414E-03 0.00885  2.66679E-04 0.01964  4.79917E-05 0.04046 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42879E-01 0.01382  1.24760E-02 2.8E-05  3.23331E-02 8.9E-05  1.06029E-01 0.00057  2.97326E-01 0.00027  1.23601E+00 0.00022  5.52688E+00 0.03725 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24504E-03 0.00788  2.60057E-04 0.02929  8.22200E-04 0.01576  5.97398E-04 0.01953  1.25405E-03 0.01317  2.64723E-04 0.02717  4.66136E-05 0.06068 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38495E-01 0.01932  1.24760E-02 3.6E-05  3.23304E-02 0.00012  1.06060E-01 0.00085  2.97240E-01 0.00037  1.23641E+00 0.00028  6.56747E+00 0.02925 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.20504E-07 0.00118  8.20162E-07 0.00119  9.29229E-07 0.01679 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14302E-07 0.00112  8.13963E-07 0.00112  9.22242E-07 0.01681 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21081E-03 0.00933  2.56936E-04 0.03066  8.15636E-04 0.01784  5.84731E-04 0.02176  1.23547E-03 0.01454  2.67571E-04 0.03064  5.04684E-05 0.06942 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.49802E-01 0.02404  1.24748E-02 6.7E-05  3.23328E-02 0.00015  1.06070E-01 0.00097  2.97361E-01 0.00042  1.23626E+00 0.00038  6.54452E+00 0.03978 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.27988E-07 0.01967  7.27568E-07 0.01967  8.26883E-07 0.06059 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22672E-07 0.01966  7.22254E-07 0.01966  8.21315E-07 0.06067 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.07114E-03 0.04389  2.29202E-04 0.13921  7.91978E-04 0.07819  5.59651E-04 0.08912  1.20956E-03 0.05785  2.36580E-04 0.12516  4.41692E-05 0.27156 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.23967E-01 0.07026  1.24764E-02 0.00012  3.23488E-02 0.00055  1.05667E-01 0.00273  2.97184E-01 0.00151  1.23673E+00 0.00099  5.36997E+00 0.13574 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.09239E-03 0.04244  2.29142E-04 0.13390  7.98199E-04 0.07647  5.58005E-04 0.08735  1.22180E-03 0.05576  2.37461E-04 0.12442  4.77909E-05 0.27407 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.24094E-01 0.07030  1.24765E-02 0.00012  3.23502E-02 0.00056  1.05648E-01 0.00264  2.97138E-01 0.00148  1.23673E+00 0.00099  5.36997E+00 0.13574 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.24054E+03 0.03947 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.14869E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08705E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22012E-03 0.00594 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95112E+03 0.00582 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.03656E-09 0.00115 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.27435E-05 0.00509  7.27429E-05 0.00509  3.96457E-06 0.24829 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.82933E-05 0.01575  7.82267E-05 0.01586  5.64882E-06 0.30202 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.87983E-04 0.01518  3.87546E-04 0.01517  5.31893E-04 0.23749 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75028E+01 0.01104 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44472E+01 0.00026  4.01513E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 09:04:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01114E+00  1.00296E+00  1.00179E+00  9.93360E-01  9.99831E-01  9.94991E-01  9.97617E-01  9.99268E-01  9.94942E-01  9.98317E-01  1.00097E+00  1.00146E+00  9.97274E-01  9.96657E-01  1.00244E+00  1.00110E+00  1.00194E+00  1.00394E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.11918E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58808E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40090E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53904E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66021E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46239E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45334E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.13016E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.74588E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333750 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33352E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33352E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.04261E+02 ;
RUNNING_TIME              (idx, 1)        =  8.30546E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.86483E-01  1.15300E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  8.16069E+01  4.88320E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.85333E-02  1.85333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.70990E+00  1.01610E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.20762E+01  5.09890E+02 ];
CPU_USAGE                 (idx, 1)        = 9.68352 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99003E+00 2.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.65875E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  5.79300E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60558E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65945E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.49689E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.05436E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26523E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46919E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.48863E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.66228E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.42549E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.84105E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06314E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17818E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44271E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35651E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17538E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55426E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59957E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38166E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.40747E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51703E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51476E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.33937E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.22463E+15 0.00039  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08077E-04 -2.38631E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.96743E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.56466E+18 0.00373  1.66809E-02 0.00370 ];
U233_FISS                 (idx, [1:   4]) = [  9.22060E+19 0.00052  9.83008E-01 6.3E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.62039E+14 0.25209  3.86037E-06 0.25211 ];
TH232_CAPT                (idx, [1:   4]) = [  9.46434E+19 0.00054  7.54319E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33985E+19 0.00131  1.06788E-01 0.00124 ];
U235_CAPT                 (idx, [1:   4]) = [  9.66617E+13 0.49753  7.68071E-07 0.49758 ];
SM149_CAPT                (idx, [1:   4]) = [  1.29777E+16 0.04727  1.03540E-04 0.04728 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000570 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14882E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000570 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5204226 5.21007E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3890863 3.89504E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905481 9.06377E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000570 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.97440E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36444E+20 4.0E-06  2.36444E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.3E-07  9.37764E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25506E+20 0.00031  1.12132E+20 0.00023  1.33733E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19282E+20 0.00017  2.05909E+20 0.00013  1.33733E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40821E+20 0.00039  2.40821E+20 0.00039  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.18035E+22 0.00028  3.01717E+22 0.00025  1.63174E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.18282E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41110E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07363E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.69011E-01 0.07935 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.97455E-01 0.01004 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.57986E-04 0.01794 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.11242E+04 0.02467 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09367E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.71953E-01 0.06569 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.29231E-01 0.06569 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52135E+00 3.9E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.81748E-01 0.00044  1.08767E-01 0.00042  3.52517E-04 0.00900 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.81791E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.81867E-01 0.00039 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.81791E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07964E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02096E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02169E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33979E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33867E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49123E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49050E-01 0.00052 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.29653E-03 0.00539  2.64459E-04 0.01929  8.02573E-04 0.01017  6.01295E-04 0.01234  1.30589E-03 0.00892  2.73225E-04 0.01982  4.90904E-05 0.04695 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45643E-01 0.01495  1.24756E-02 3.3E-05  3.23349E-02 8.9E-05  1.06020E-01 0.00060  2.97405E-01 0.00025  1.23545E+00 0.00022  5.31380E+00 0.04033 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.20958E-03 0.00808  2.54069E-04 0.02951  7.82013E-04 0.01669  5.95406E-04 0.01813  1.26803E-03 0.01244  2.63961E-04 0.02951  4.60965E-05 0.06879 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.37834E-01 0.02107  1.24760E-02 3.7E-05  3.23317E-02 0.00012  1.05977E-01 0.00075  2.97451E-01 0.00037  1.23556E+00 0.00030  6.59574E+00 0.03038 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.29067E-07 0.00119  8.28775E-07 0.00119  9.19459E-07 0.01609 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.13897E-07 0.00113  8.13610E-07 0.00114  9.02746E-07 0.01613 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22218E-03 0.00907  2.62020E-04 0.02891  7.86051E-04 0.01741  5.92954E-04 0.02112  1.26525E-03 0.01511  2.65634E-04 0.03277  5.02722E-05 0.07199 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.48164E-01 0.02507  1.24761E-02 4.0E-05  3.23333E-02 0.00015  1.06111E-01 0.00111  2.97470E-01 0.00045  1.23518E+00 0.00039  6.48158E+00 0.04099 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.33795E-07 0.01955  7.33573E-07 0.01955  8.11622E-07 0.05651 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20420E-07 0.01955  7.20204E-07 0.01955  7.96365E-07 0.05645 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.75712E-03 0.04450  2.05806E-04 0.13262  7.49952E-04 0.07924  5.47152E-04 0.09580  1.00725E-03 0.06140  2.06085E-04 0.13485  4.08769E-05 0.29808 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.10912E-01 0.09844  1.24757E-02 0.00015  3.23089E-02 0.00039  1.06111E-01 0.00336  2.97028E-01 0.00155  1.23428E+00 0.00118  6.14486E+00 0.13882 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.77914E-03 0.04360  2.08477E-04 0.13166  7.53677E-04 0.07694  5.61361E-04 0.09477  1.01214E-03 0.06047  2.04010E-04 0.13047  3.94720E-05 0.29083 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.10083E-01 0.09941  1.24758E-02 0.00014  3.23080E-02 0.00038  1.06094E-01 0.00331  2.97103E-01 0.00154  1.23424E+00 0.00118  6.14486E+00 0.13882 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75427E+03 0.03973 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.24008E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08924E-07 0.00047 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21483E-03 0.00596 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.90208E+03 0.00602 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99852E-09 0.00121 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.17765E-05 0.00514  7.17678E-05 0.00514  1.49649E-06 0.40899 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.42415E-05 0.01665  7.42395E-05 0.01665  1.72700E-06 0.46355 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.53279E-04 0.01748  3.53746E-04 0.01753  1.91717E-04 0.42099 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69501E+01 0.01196 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45334E+01 0.00027  4.01865E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 09:56:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00323E+00  1.00223E+00  9.98492E-01  1.00207E+00  9.97855E-01  9.95191E-01  9.96983E-01  1.00290E+00  9.98757E-01  1.00163E+00  1.00007E+00  1.00140E+00  1.00012E+00  9.95269E-01  1.00181E+00  9.98056E-01  1.00240E+00  1.00153E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.09608E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59039E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40533E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54271E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65612E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47625E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46722E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.14243E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.73342E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334143 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33410E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33410E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.31054E+03 ;
RUNNING_TIME              (idx, 1)        =  1.34795E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.09567E-01  1.23083E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.33195E+02  5.15880E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.71000E-02  1.85667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.81052E+00  1.10055E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.33733E+02  7.21417E+02 ];
CPU_USAGE                 (idx, 1)        = 9.72243 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99028E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.69981E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.42139E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65780E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.24114E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88879E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32008E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.50141E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49454E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.81913E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.41433E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.91954E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.11282E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89938E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60305E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49777E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34972E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33626E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54699E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.69922E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.18134E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.03529E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51970E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.33378E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.57948E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.31414E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.73204E-04 -1.26562E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.11350E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58533E+18 0.00427  1.68938E-02 0.00423 ];
U233_FISS                 (idx, [1:   4]) = [  9.21305E+19 0.00047  9.81792E-01 7.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.04882E+16 0.04972  1.11778E-04 0.04973 ];
TH232_CAPT                (idx, [1:   4]) = [  9.59022E+19 0.00053  7.48033E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34449E+19 0.00137  1.04869E-01 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  3.56069E+15 0.08553  2.77794E-05 0.08548 ];
SM149_CAPT                (idx, [1:   4]) = [  5.84140E+16 0.01950  4.55678E-04 0.01951 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002309 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16036E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002309 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5253509 5.25857E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3845516 3.84899E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 903284 9.04040E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002309 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.86847E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36435E+20 3.7E-06  2.36435E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37759E+19 1.2E-07  9.37759E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28228E+20 0.00029  1.14584E+20 0.00021  1.36446E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22004E+20 0.00017  2.08359E+20 0.00011  1.36446E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43805E+20 0.00036  2.43805E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.22339E+22 0.00025  3.05727E+22 0.00022  1.66119E+21 0.00171 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20418E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.44046E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09033E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51427E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51427E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.53923E-01 0.05634 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.01029E-01 0.01229 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.24165E-04 0.02347 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.01963E+04 0.02446 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09600E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.93458E-01 0.04244 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.30856E-01 0.04244 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52128E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.70341E-01 0.00042  1.07483E-01 0.00041  3.43592E-04 0.00892 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.69948E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.69809E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.69948E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06637E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02513E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02558E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33425E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33345E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50246E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50130E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33488E-03 0.00594  2.63751E-04 0.02131  8.34308E-04 0.01152  6.19810E-04 0.01205  1.29198E-03 0.00845  2.73691E-04 0.02076  5.13396E-05 0.04267 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.41656E-01 0.01474  1.24759E-02 2.8E-05  3.23330E-02 9.2E-05  1.06062E-01 0.00057  2.97422E-01 0.00026  1.23625E+00 0.00023  5.12338E+00 0.03812 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.24139E-03 0.00834  2.52622E-04 0.03117  8.21600E-04 0.01649  5.93612E-04 0.01867  1.26300E-03 0.01233  2.60928E-04 0.02959  4.96353E-05 0.06120 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.39277E-01 0.02012  1.24758E-02 3.6E-05  3.23322E-02 0.00013  1.06127E-01 0.00088  2.97412E-01 0.00038  1.23655E+00 0.00030  6.18644E+00 0.03025 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.41256E-07 0.00123  8.40907E-07 0.00123  9.49425E-07 0.01697 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16274E-07 0.00120  8.15936E-07 0.00120  9.21246E-07 0.01699 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19237E-03 0.00928  2.43764E-04 0.03339  7.99901E-04 0.01791  5.96396E-04 0.02032  1.23767E-03 0.01494  2.70404E-04 0.03341  4.42271E-05 0.07548 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.39753E-01 0.02432  1.24758E-02 5.0E-05  3.23435E-02 0.00016  1.06088E-01 0.00098  2.97433E-01 0.00044  1.23656E+00 0.00046  6.46126E+00 0.04292 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.45924E-07 0.01958  7.45530E-07 0.01958  8.53601E-07 0.06266 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.24031E-07 0.01957  7.23648E-07 0.01957  8.28357E-07 0.06251 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83684E-03 0.04205  2.20966E-04 0.13251  7.09903E-04 0.07342  5.14338E-04 0.09037  1.12636E-03 0.06754  2.33782E-04 0.14126  3.14842E-05 0.38041 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.26699E-01 0.08874  1.24761E-02 0.00013  3.23792E-02 0.00069  1.06122E-01 0.00318  2.96415E-01 0.00122  1.23685E+00 0.00104  7.91215E+00 0.11040 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85322E-03 0.04148  2.12798E-04 0.12845  7.13595E-04 0.07097  5.31552E-04 0.08843  1.12469E-03 0.06639  2.37236E-04 0.14139  3.33511E-05 0.38721 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.24824E-01 0.08725  1.24762E-02 0.00013  3.23769E-02 0.00068  1.06093E-01 0.00314  2.96376E-01 0.00119  1.23687E+00 0.00104  7.91215E+00 0.11040 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.83112E+03 0.03761 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.34164E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09379E-07 0.00053 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.16427E-03 0.00704 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.79445E+03 0.00715 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.84223E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.96758E-05 0.00676  6.96143E-05 0.00676  2.48927E-06 0.34039 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.73708E-05 0.02468  6.74269E-05 0.02480  1.96837E-06 0.39719 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.18056E-04 0.02255  2.17785E-04 0.02271  3.05344E-04 0.34190 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70947E+01 0.01168 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46722E+01 0.00026  4.03335E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 10:48:32 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00934E+00  9.99151E-01  9.98465E-01  1.00125E+00  9.99382E-01  9.98123E-01  9.92689E-01  1.00246E+00  1.00075E+00  9.98975E-01  1.00854E+00  9.92591E-01  1.00410E+00  1.00038E+00  1.00259E+00  9.93390E-01  1.00057E+00  9.97251E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.08069E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59193E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40494E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54192E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65003E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48502E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47598E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16121E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.72688E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334209 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33400E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33400E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.82220E+03 ;
RUNNING_TIME              (idx, 1)        =  1.87025E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.40850E-01  1.31283E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.85263E+02  5.20683E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.50167E-02  1.79167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.85352E+00  1.04295E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.86024E+02  7.55133E+02 ];
CPU_USAGE                 (idx, 1)        = 9.74307 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99045E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.72186E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.53734E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67404E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.16106E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92327E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34325E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58251E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50810E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.87058E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.57577E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.07012E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.29444E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79936E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74630E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48907E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36812E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35044E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56605E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39970E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42654E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10760E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55050E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.17057E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.73595E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.36647E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.11965E-03 -2.47216E+26  2.21044E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.25281E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58180E+18 0.00389  1.68767E-02 0.00381 ];
U233_FISS                 (idx, [1:   4]) = [  9.18764E+19 0.00052  9.80302E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.16598E+16 0.02333  4.44755E-04 0.02340 ];
TH232_CAPT                (idx, [1:   4]) = [  9.68045E+19 0.00052  7.45294E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33877E+19 0.00140  1.03071E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  1.41218E+16 0.04188  1.08715E-04 0.04191 ];
SM149_CAPT                (idx, [1:   4]) = [  9.91562E+16 0.01598  7.63338E-04 0.01596 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002004 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13944E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002004 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5284559 5.28973E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3813365 3.81687E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904080 9.04790E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002004 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.67524E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36427E+20 3.7E-06  2.36427E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 1.2E-07  9.37751E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29961E+20 0.00032  1.16216E+20 0.00022  1.37442E+19 0.00202 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23736E+20 0.00019  2.09992E+20 0.00012  1.37442E+19 0.00202 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.45549E+20 0.00035  2.45549E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24932E+22 0.00025  3.08245E+22 0.00022  1.66866E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.22176E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.45953E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10027E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51427E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51427E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.99548E-01 0.06220 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.32865E-01 0.01352 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.72486E-04 0.02555 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.51806E+03 0.02933 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09524E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.41839E-01 0.04657 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.83765E-01 0.04657 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52121E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.62065E-01 0.00044  1.06583E-01 0.00042  3.41344E-04 0.00981 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.62388E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.62885E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.62388E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05807E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02601E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02887E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33306E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32907E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50888E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50867E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.37418E-03 0.00553  2.69962E-04 0.01911  8.18020E-04 0.01029  6.16436E-04 0.01303  1.33110E-03 0.00888  2.82301E-04 0.01842  5.63698E-05 0.04761 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.53655E-01 0.01513  1.24754E-02 2.9E-05  3.23356E-02 9.3E-05  1.06068E-01 0.00058  2.97482E-01 0.00027  1.23567E+00 0.00023  5.18928E+00 0.03774 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22084E-03 0.00815  2.58436E-04 0.02741  7.82652E-04 0.01523  5.91099E-04 0.01995  1.26879E-03 0.01164  2.68255E-04 0.02950  5.16019E-05 0.06540 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.49222E-01 0.02138  1.24754E-02 3.8E-05  3.23304E-02 0.00012  1.06116E-01 0.00092  2.97339E-01 0.00038  1.23526E+00 0.00032  6.28308E+00 0.02887 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.47025E-07 0.00129  8.46693E-07 0.00130  9.50047E-07 0.01490 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14835E-07 0.00118  8.14516E-07 0.00118  9.13868E-07 0.01488 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18526E-03 0.00992  2.44051E-04 0.03217  7.84370E-04 0.01811  5.82207E-04 0.02003  1.25721E-03 0.01525  2.67222E-04 0.03311  5.01977E-05 0.07412 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.46860E-01 0.02527  1.24760E-02 4.2E-05  3.23336E-02 0.00016  1.06265E-01 0.00103  2.97498E-01 0.00044  1.23584E+00 0.00041  6.15543E+00 0.04297 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.51083E-07 0.01955  7.50702E-07 0.01956  8.41888E-07 0.05080 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22798E-07 0.01954  7.22433E-07 0.01954  8.09918E-07 0.05067 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98931E-03 0.04519  1.85446E-04 0.14919  6.74958E-04 0.07882  6.15490E-04 0.08806  1.16163E-03 0.06922  3.21899E-04 0.11598  2.98835E-05 0.30165 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.47503E-01 0.08484  1.24745E-02 0.00017  3.23405E-02 0.00057  1.06592E-01 0.00356  2.97475E-01 0.00162  1.23555E+00 0.00105  6.25117E+00 0.14173 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98555E-03 0.04427  1.88140E-04 0.14639  6.72921E-04 0.07708  6.11129E-04 0.08562  1.15480E-03 0.06763  3.27403E-04 0.11210  3.11574E-05 0.29278 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.56213E-01 0.08670  1.24745E-02 0.00017  3.23398E-02 0.00056  1.06596E-01 0.00355  2.97430E-01 0.00160  1.23557E+00 0.00105  6.30445E+00 0.14069 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.99002E+03 0.04068 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.41920E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09934E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.26018E-03 0.00567 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.87274E+03 0.00570 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78958E-09 0.00083 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92406E-05 0.00920  6.91900E-05 0.00925  7.96580E-07 0.58449 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.44669E-05 0.03061  6.44950E-05 0.03074  3.26061E-07 0.68824 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.71480E-04 0.02434  1.71763E-04 0.02444  8.69491E-05 0.57633 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72724E+01 0.01107 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47598E+01 0.00028  4.04090E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 11:42:40 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01374E+00  1.00447E+00  1.00001E+00  1.00076E+00  9.92846E-01  9.96398E-01  9.97956E-01  9.97765E-01  9.96520E-01  1.00512E+00  9.99313E-01  9.95536E-01  9.96584E-01  9.95609E-01  1.00357E+00  1.00314E+00  9.96814E-01  1.00384E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.07183E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59282E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40632E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54310E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64410E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49806E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48903E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18081E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.72506E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334321 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33405E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33405E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.34432E+03 ;
RUNNING_TIME              (idx, 1)        =  2.41158E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.75533E-01  1.34683E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.39226E+02  5.39633E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.35667E-02  1.85500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.83037E+00  1.97678E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.39225E+02  7.61439E+02 ];
CPU_USAGE                 (idx, 1)        = 9.72107 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98907E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.69750E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.56783E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67037E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.51316E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94342E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35568E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.59267E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50303E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00547E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.69904E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.33942E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.38602E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.70952E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.86032E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93958E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33795E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32733E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53414E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35566E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81749E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10803E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52881E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51118E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.78387E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.45717E+15 0.00038  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.26149E-03 -4.99330E+26  2.21297E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.44880E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.60091E+18 0.00404  1.70573E-02 0.00398 ];
U233_FISS                 (idx, [1:   4]) = [  9.16771E+19 0.00055  9.76825E-01 8.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.57205E+17 0.01265  1.67520E-03 0.01266 ];
TH232_CAPT                (idx, [1:   4]) = [  9.81671E+19 0.00051  7.40584E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33482E+19 0.00133  1.00703E-01 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  5.58047E+16 0.02075  4.20927E-04 0.02072 ];
SM149_CAPT                (idx, [1:   4]) = [  1.34457E+17 0.01371  1.01440E-03 0.01370 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002156 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15878E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002156 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5327476 5.33265E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3772159 3.77565E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 902521 9.03289E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002156 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.79397E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36417E+20 3.8E-06  2.36417E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37730E+19 1.3E-07  9.37730E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32577E+20 0.00030  1.18631E+20 0.00021  1.39459E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26350E+20 0.00018  2.12404E+20 0.00011  1.39459E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48572E+20 0.00038  2.48572E+20 0.00038  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29481E+22 0.00028  3.12546E+22 0.00024  1.69343E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.24542E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.48804E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11707E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51144E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51144E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.37272E-01 0.06110 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.11378E-01 0.01352 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.58252E-04 0.02834 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.32983E+03 0.02985 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09674E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.20792E-01 0.04788 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.64662E-01 0.04788 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52116E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.51740E-01 0.00043  1.05432E-01 0.00041  3.37410E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.51325E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.51140E-01 0.00038 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.51325E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04581E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02357E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02574E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33635E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33326E-02 0.00100 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53839E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53369E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.39961E-03 0.00585  2.71975E-04 0.01962  8.35348E-04 0.01108  6.22644E-04 0.01400  1.33113E-03 0.00884  2.83720E-04 0.01961  5.47915E-05 0.04104 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.52117E-01 0.01356  1.24758E-02 2.8E-05  3.23299E-02 9.3E-05  1.06310E-01 0.00061  2.97440E-01 0.00027  1.23618E+00 0.00033  5.53173E+00 0.03721 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22883E-03 0.00805  2.62811E-04 0.02733  7.90870E-04 0.01515  5.85255E-04 0.02040  1.26825E-03 0.01300  2.69346E-04 0.02959  5.22946E-05 0.06429 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.52723E-01 0.02156  1.24760E-02 3.4E-05  3.23280E-02 0.00013  1.06317E-01 0.00086  2.97456E-01 0.00039  1.23650E+00 0.00048  6.58715E+00 0.02884 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.57511E-07 0.00124  8.57244E-07 0.00124  9.42171E-07 0.01701 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16067E-07 0.00110  8.15813E-07 0.00110  8.96640E-07 0.01700 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18335E-03 0.00904  2.61882E-04 0.03240  7.72691E-04 0.01934  5.88998E-04 0.02324  1.24033E-03 0.01443  2.67867E-04 0.03068  5.15909E-05 0.07161 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.57898E-01 0.02515  1.24760E-02 5.1E-05  3.23295E-02 0.00018  1.06333E-01 0.00106  2.97490E-01 0.00047  1.23548E+00 0.00061  6.78442E+00 0.03992 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.58871E-07 0.01963  7.58443E-07 0.01963  8.86498E-07 0.07427 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22688E-07 0.01962  7.22281E-07 0.01962  8.44639E-07 0.07439 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.62975E-03 0.04576  2.24763E-04 0.13826  6.53886E-04 0.08015  5.10049E-04 0.10319  9.75317E-04 0.06989  2.08364E-04 0.14847  5.73676E-05 0.26910 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.53016E-01 0.09522  1.24747E-02 0.00016  3.22877E-02 0.00034  1.06454E-01 0.00351  2.97812E-01 0.00180  1.23465E+00 0.00129  5.81117E+00 0.12524 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.65111E-03 0.04523  2.29792E-04 0.14099  6.44158E-04 0.07808  5.18440E-04 0.09870  9.93455E-04 0.06939  2.11210E-04 0.14572  5.40535E-05 0.27148 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52989E-01 0.09308  1.24749E-02 0.00015  3.22877E-02 0.00034  1.06447E-01 0.00350  2.97870E-01 0.00180  1.23469E+00 0.00128  5.81117E+00 0.12524 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.49878E+03 0.04211 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.51414E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.10272E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.13800E-03 0.00564 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.68677E+03 0.00576 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77020E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69039E-05 0.01034  6.68165E-05 0.01000  1.15151E-06 0.60902 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.65964E-05 0.02958  5.66088E-05 0.02967  7.79841E-07 0.67815 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.58227E-04 0.02696  1.58364E-04 0.02694  1.25807E-04 0.50048 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70293E+01 0.01169 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48903E+01 0.00028  4.04823E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 12:37:05 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01184E+00  1.00118E+00  1.00146E+00  1.00296E+00  9.98738E-01  9.95303E-01  9.94764E-01  1.00203E+00  1.00064E+00  1.00160E+00  9.89728E-01  9.98463E-01  1.00378E+00  1.00152E+00  9.96386E-01  9.92427E-01  1.00367E+00  1.00351E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.07237E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59276E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40366E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54051E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64336E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50430E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49524E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20142E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.73142E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334607 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33403E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33403E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.85955E+03 ;
RUNNING_TIME              (idx, 1)        =  2.95580E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.04767E-01  1.29233E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.93481E+02  5.42543E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.32667E-02  1.97000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  8.90353E+00  3.07308E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.92543E+02  7.79553E+02 ];
CPU_USAGE                 (idx, 1)        = 9.67435 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99003E+00 2.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.64520E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.62886E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67899E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.16783E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97291E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37490E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62388E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50942E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10178E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.80643E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.73772E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.52321E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.26534E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.95382E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35659E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.34685E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33395E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54300E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.03180E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17772E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14204E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55879E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.33725E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.87175E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.52922E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.38823E-03 -7.48111E+26  2.21545E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.62219E-01 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.61338E+18 0.00395  1.71844E-02 0.00389 ];
U233_FISS                 (idx, [1:   4]) = [  9.13359E+19 0.00050  9.72868E-01 8.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.33566E+17 0.00878  3.55314E-03 0.00878 ];
TH232_CAPT                (idx, [1:   4]) = [  9.92784E+19 0.00052  7.37396E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33267E+19 0.00135  9.89851E-02 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  1.17334E+17 0.01497  8.71394E-04 0.01492 ];
SM149_CAPT                (idx, [1:   4]) = [  1.55816E+17 0.01249  1.15744E-03 0.01251 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002089 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16307E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002089 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5359149 5.36448E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3737300 3.74079E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905640 9.06364E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002089 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.75671E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36406E+20 3.5E-06  2.36406E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37701E+19 1.1E-07  9.37701E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34615E+20 0.00030  1.20563E+20 0.00021  1.40518E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.28385E+20 0.00018  2.14333E+20 0.00012  1.40518E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.50974E+20 0.00036  2.50974E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.33071E+22 0.00026  3.16043E+22 0.00023  1.70274E+21 0.00186 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.27479E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.51133E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12943E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50737E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50737E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.81715E-01 0.05670 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19690E-01 0.01476 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.47777E-04 0.02641 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.72055E+03 0.02635 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09366E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.53255E-01 0.04432 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.93989E-01 0.04432 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52112E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.43027E-01 0.00042  1.04441E-01 0.00042  3.43582E-04 0.00907 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.42461E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.41991E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.42461E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03645E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02336E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02144E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33659E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33900E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54935E-01 0.00173 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.55931E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49731E-03 0.00572  2.79658E-04 0.01997  8.46578E-04 0.01100  6.36826E-04 0.01278  1.38010E-03 0.00919  2.98095E-04 0.01936  5.60540E-05 0.04197 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.55668E-01 0.01514  1.24754E-02 2.9E-05  3.23348E-02 0.00011  1.06229E-01 0.00059  2.97683E-01 0.00027  1.23712E+00 0.00030  5.50766E+00 0.03615 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28227E-03 0.00884  2.56114E-04 0.02926  7.85228E-04 0.01681  5.81665E-04 0.01857  1.32632E-03 0.01368  2.83406E-04 0.02665  4.95383E-05 0.05577 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.54097E-01 0.01869  1.24755E-02 3.9E-05  3.23314E-02 0.00014  1.06336E-01 0.00093  2.97818E-01 0.00039  1.23759E+00 0.00048  6.50509E+00 0.02853 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.67349E-07 0.00132  8.66898E-07 0.00131  9.92317E-07 0.02416 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17883E-07 0.00123  8.17458E-07 0.00122  9.35524E-07 0.02394 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27734E-03 0.00924  2.62159E-04 0.03275  7.88148E-04 0.01827  6.07546E-04 0.02167  1.29561E-03 0.01444  2.74320E-04 0.03244  4.95533E-05 0.07339 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.46002E-01 0.02502  1.24755E-02 5.0E-05  3.23320E-02 0.00017  1.06255E-01 0.00107  2.97652E-01 0.00044  1.23883E+00 0.00054  6.34142E+00 0.04204 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.67404E-07 0.01975  7.67136E-07 0.01975  8.36665E-07 0.05695 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23889E-07 0.01974  7.23636E-07 0.01974  7.89002E-07 0.05688 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.78231E-03 0.04514  2.52940E-04 0.13921  6.16785E-04 0.08169  5.05730E-04 0.09657  1.13271E-03 0.06358  2.29960E-04 0.13936  4.41885E-05 0.29309 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.31473E-01 0.08427  1.24748E-02 0.00016  3.23283E-02 0.00054  1.06053E-01 0.00315  2.98066E-01 0.00175  1.23730E+00 0.00111  6.52551E+00 0.14166 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.80298E-03 0.04456  2.54650E-04 0.13518  6.30548E-04 0.08012  5.04203E-04 0.09534  1.14438E-03 0.06348  2.25902E-04 0.14029  4.32993E-05 0.30562 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.24843E-01 0.08213  1.24750E-02 0.00016  3.23307E-02 0.00056  1.06074E-01 0.00316  2.98100E-01 0.00175  1.23725E+00 0.00113  6.52551E+00 0.14166 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65606E+03 0.04123 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.60858E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11772E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29518E-03 0.00657 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82939E+03 0.00674 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77849E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71011E-05 0.00811  6.70460E-05 0.00820  2.74390E-06 0.32439 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.21246E-05 0.03045  6.18644E-05 0.03063  2.77931E-06 0.42288 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.47255E-04 0.02500  1.46772E-04 0.02514  2.92878E-04 0.31246 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67691E+01 0.01120 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49524E+01 0.00026  4.05029E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 13:32:37 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01187E+00  9.98769E-01  1.00394E+00  9.99210E-01  1.00226E+00  9.99137E-01  9.95139E-01  1.00626E+00  9.98382E-01  1.00253E+00  9.95698E-01  9.99533E-01  9.93180E-01  9.99254E-01  1.00190E+00  9.94184E-01  1.00020E+00  9.98558E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.07181E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59282E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40064E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53752E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64353E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51542E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50634E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23239E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.74027E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334508 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33420E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33420E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.38534E+03 ;
RUNNING_TIME              (idx, 1)        =  3.51112E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.38133E-01  1.33367E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.48838E+02  5.53573E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.12483E-01  1.92167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.20296E+01  3.12605E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.48023E+02  7.83278E+02 ];
CPU_USAGE                 (idx, 1)        = 9.64178 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99057E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.60879E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.65657E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67832E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.13964E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99487E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38746E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62921E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50709E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27308E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.94134E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.72925E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.65382E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.95592E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.07506E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07986E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32435E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31590E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51851E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.85855E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.79625E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14722E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.57223E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.91251E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.91148E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.62449E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43284E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.71370E-03 -1.26157E+27  2.22059E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.89353E-01 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.61797E+18 0.00424  1.72580E-02 0.00419 ];
U233_FISS                 (idx, [1:   4]) = [  9.03523E+19 0.00051  9.63755E-01 0.00010 ];
U235_FISS                 (idx, [1:   4]) = [  8.20068E+17 0.00526  8.74778E-03 0.00527 ];
PU239_FISS                (idx, [1:   4]) = [  2.52223E+13 1.00000  2.69491E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00594E+20 0.00052  7.31141E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32077E+19 0.00130  9.59990E-02 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  2.88763E+17 0.00925  2.09878E-03 0.00924 ];
SM149_CAPT                (idx, [1:   4]) = [  1.61410E+17 0.01210  1.17323E-03 0.01211 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002609 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14885E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002609 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5408616 5.41355E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3685531 3.68880E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908462 9.09136E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002609 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.42144E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36374E+20 3.8E-06  2.36374E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37628E+19 1.1E-07  9.37628E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37613E+20 0.00029  1.23343E+20 0.00019  1.42702E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31376E+20 0.00017  2.17105E+20 0.00011  1.42702E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.54150E+20 0.00037  2.54150E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.38041E+22 0.00026  3.20791E+22 0.00022  1.72499E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.31064E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.54482E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14654E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49923E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49923E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.85168E-01 0.05749 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.26360E-01 0.01380 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.55106E-04 0.02579 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.27120E+03 0.02781 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09089E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.58148E-01 0.04306 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.98362E-01 0.04306 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52098E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.29878E-01 0.00042  1.02988E-01 0.00040  3.38213E-04 0.00897 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.29928E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.30095E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.29928E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02291E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01424E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01528E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34887E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34726E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.60110E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.60392E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.55636E-03 0.00538  2.93050E-04 0.02007  8.67003E-04 0.01113  6.50879E-04 0.01289  1.39691E-03 0.00879  2.91719E-04 0.01867  5.68028E-05 0.04424 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.45558E-01 0.01458  1.24756E-02 2.8E-05  3.23140E-02 9.7E-05  1.06298E-01 0.00059  2.98043E-01 0.00029  1.23957E+00 0.00036  5.14337E+00 0.03764 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28418E-03 0.00790  2.68345E-04 0.02695  8.06456E-04 0.01652  6.05319E-04 0.01925  1.28269E-03 0.01352  2.66919E-04 0.02918  5.44557E-05 0.06381 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.51821E-01 0.02220  1.24754E-02 3.9E-05  3.23129E-02 0.00016  1.06271E-01 0.00078  2.98004E-01 0.00038  1.24046E+00 0.00056  6.33512E+00 0.02870 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.78527E-07 0.00121  8.78115E-07 0.00121  1.00402E-06 0.02520 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16882E-07 0.00114  8.16499E-07 0.00115  9.33501E-07 0.02507 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.27562E-03 0.00916  2.70058E-04 0.03100  7.98305E-04 0.01741  5.96160E-04 0.02160  1.28292E-03 0.01447  2.74903E-04 0.03281  5.32706E-05 0.06845 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51968E-01 0.02354  1.24757E-02 4.9E-05  3.23092E-02 0.00017  1.06473E-01 0.00107  2.98160E-01 0.00048  1.23930E+00 0.00069  6.39200E+00 0.04035 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.78573E-07 0.01962  7.78166E-07 0.01962  9.05917E-07 0.05297 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.24165E-07 0.01961  7.23785E-07 0.01962  8.43118E-07 0.05303 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.90757E-03 0.04370  2.05138E-04 0.13666  6.84572E-04 0.07745  5.27004E-04 0.09887  1.20102E-03 0.06494  2.46561E-04 0.12538  4.32820E-05 0.29698 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.41190E-01 0.09286  1.24741E-02 0.00018  3.23005E-02 0.00061  1.05970E-01 0.00306  2.96745E-01 0.00140  1.23724E+00 0.00212  5.60108E+00 0.15595 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90377E-03 0.04284  1.96006E-04 0.13364  6.85307E-04 0.07668  5.17764E-04 0.09631  1.22646E-03 0.06398  2.38605E-04 0.12509  3.96234E-05 0.31892 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.32447E-01 0.08766  1.24742E-02 0.00018  3.23013E-02 0.00061  1.05954E-01 0.00304  2.96815E-01 0.00140  1.23728E+00 0.00211  5.60108E+00 0.15595 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75941E+03 0.03967 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.73950E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12623E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.34286E-03 0.00602 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82574E+03 0.00609 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.80703E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71189E-05 0.00943  6.70711E-05 0.00947  1.13248E-06 0.44877 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.17326E-05 0.03204  6.16405E-05 0.03216  7.92015E-07 0.66407 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.55293E-04 0.02486  1.55349E-04 0.02501  1.45143E-04 0.44508 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70783E+01 0.01154 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50634E+01 0.00027  4.04804E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 14:28:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01263E+00  1.00033E+00  9.94162E-01  9.96696E-01  9.97195E-01  9.96965E-01  9.95363E-01  9.99263E-01  9.98920E-01  1.00576E+00  1.00484E+00  9.99033E-01  1.00010E+00  1.00035E+00  1.00341E+00  9.95990E-01  9.98979E-01  1.00002E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.07888E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59211E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39463E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53186E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64317E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51086E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50176E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24444E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.74706E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333903 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33393E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33393E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.91880E+03 ;
RUNNING_TIME              (idx, 1)        =  4.07454E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.07750E+00  1.39367E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.04999E+02  5.61605E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.31350E-01  1.88667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.52047E+01  3.17502E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.04320E+02  7.92929E+02 ];
CPU_USAGE                 (idx, 1)        = 9.61777 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99079E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.58195E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.73256E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69183E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.00005E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01838E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39826E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.68107E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51896E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75543E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.24268E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.13457E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.88301E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.25719E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.35117E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.33215E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.32366E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31223E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51503E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.59574E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.28231E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.18235E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.67757E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84867E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98864E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.72913E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.15641E-02 -2.55331E+27  2.23351E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01833E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.63197E+18 0.00399  1.74013E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  8.81717E+19 0.00052  9.40165E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.27395E+18 0.00363  2.42462E-02 0.00356 ];
PU239_FISS                (idx, [1:   4]) = [  9.76983E+14 0.16065  1.03980E-05 0.16070 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00973E+20 0.00054  7.17854E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28808E+19 0.00138  9.15763E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  8.06011E+17 0.00569  5.73038E-03 0.00570 ];
U238_CAPT                 (idx, [1:   4]) = [  2.56683E+13 1.00000  1.82839E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  4.65102E+14 0.24231  3.30394E-06 0.24248 ];
PU240_CAPT                (idx, [1:   4]) = [  5.13854E+13 0.70599  3.63825E-07 0.70609 ];
SM149_CAPT                (idx, [1:   4]) = [  1.68213E+17 0.01201  1.19597E-03 0.01202 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001789 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12845E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001789 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5454325 5.45961E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3636728 3.64016E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910736 9.11521E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001789 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.21655E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36283E+20 3.9E-06  2.36283E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37411E+19 1.1E-07  9.37411E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40668E+20 0.00029  1.26325E+20 0.00020  1.43433E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34409E+20 0.00017  2.20066E+20 0.00012  1.43433E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57638E+20 0.00035  2.57638E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.42807E+22 0.00025  3.25514E+22 0.00022  1.72937E+21 0.00194 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.34846E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.57894E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16108E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47789E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47789E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.51317E-01 0.05737 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.42507E-01 0.01338 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.45723E-04 0.02880 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.13243E+03 0.02858 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08850E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.19548E-01 0.04592 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.63194E-01 0.04592 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52059E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99747E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.17425E-01 0.00044  1.01611E-01 0.00043  3.36601E-04 0.00885 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.17259E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.17146E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.17259E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00927E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.98063E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  6.98326E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.39492E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.39109E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.70406E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.70245E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.65226E-03 0.00596  2.83643E-04 0.02018  8.74929E-04 0.01124  6.76206E-04 0.01212  1.43086E-03 0.00943  3.17925E-04 0.01838  6.87015E-05 0.03798 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.75256E-01 0.01367  1.24756E-02 3.0E-05  3.22956E-02 0.00011  1.06300E-01 0.00056  2.98913E-01 0.00029  1.24588E+00 0.00055  5.63449E+00 0.03217 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26223E-03 0.00795  2.56359E-04 0.02891  7.76049E-04 0.01596  6.10134E-04 0.01856  1.27372E-03 0.01318  2.87186E-04 0.02656  5.87754E-05 0.05761 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.73530E-01 0.01992  1.24753E-02 4.4E-05  3.23003E-02 0.00017  1.06300E-01 0.00080  2.98975E-01 0.00043  1.24601E+00 0.00079  6.59938E+00 0.02496 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.82812E-07 0.00123  8.82516E-07 0.00124  9.69578E-07 0.01527 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.09869E-07 0.00116  8.09598E-07 0.00117  8.89461E-07 0.01525 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.29771E-03 0.00880  2.59452E-04 0.03477  7.83991E-04 0.01912  6.19277E-04 0.02178  1.29808E-03 0.01405  2.78647E-04 0.03190  5.82610E-05 0.06622 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.62876E-01 0.02401  1.24760E-02 5.0E-05  3.22929E-02 0.00018  1.06425E-01 0.00106  2.99042E-01 0.00049  1.24527E+00 0.00106  6.43928E+00 0.03816 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.79892E-07 0.01964  7.79551E-07 0.01965  8.16769E-07 0.06077 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.15680E-07 0.01964  7.15366E-07 0.01964  7.49802E-07 0.06083 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08766E-03 0.04248  2.14977E-04 0.14207  7.56870E-04 0.08202  6.22561E-04 0.08567  1.20026E-03 0.06205  2.34045E-04 0.12981  5.89381E-05 0.29289 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.61113E-01 0.10066  1.24762E-02 0.00014  3.23379E-02 0.00070  1.06100E-01 0.00298  3.00110E-01 0.00195  1.25412E+00 0.00349  7.24377E+00 0.09479 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.07286E-03 0.04226  2.15417E-04 0.13897  7.45278E-04 0.08244  6.22696E-04 0.08430  1.19603E-03 0.06042  2.35934E-04 0.13316  5.75030E-05 0.27272 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.69981E-01 0.09759  1.24762E-02 0.00014  3.23375E-02 0.00070  1.06110E-01 0.00299  2.99963E-01 0.00190  1.25424E+00 0.00347  7.24377E+00 0.09479 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.97656E+03 0.03787 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.76010E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.03625E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35346E-03 0.00610 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82869E+03 0.00614 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.80205E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.69043E-05 0.00906  6.68938E-05 0.00908  1.18933E-06 0.46194 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88340E-05 0.03015  5.89611E-05 0.03016  4.87788E-07 0.51210 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.45045E-04 0.02703  1.45072E-04 0.02708  1.46409E-04 0.44486 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68781E+01 0.01122 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50176E+01 0.00026  4.02097E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 15:25:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01355E+00  9.99357E-01  9.99147E-01  9.98814E-01  9.97745E-01  9.94071E-01  1.00249E+00  9.99431E-01  9.93605E-01  9.99000E-01  9.96854E-01  9.99147E-01  9.95590E-01  1.00646E+00  1.00130E+00  1.00498E+00  9.94644E-01  1.00381E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.08937E-02 0.00030  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59106E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39955E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53693E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64220E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50053E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49146E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20734E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.74599E+00 0.00040  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334422 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33428E+04 0.00071 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33428E+04 0.00071 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.45817E+03 ;
RUNNING_TIME              (idx, 1)        =  4.64326E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.21747E+00  1.39967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.61685E+02  5.66860E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.51150E-01  1.98000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.83022E+01  3.09743E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.61266E+02  7.99188E+02 ];
CPU_USAGE                 (idx, 1)        = 9.60140 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99074E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.56373E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.87972E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73205E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.50355E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01985E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39337E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82607E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55905E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.10522E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78027E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.02266E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20882E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04719E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85241E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38570E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38533E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35376E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57500E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.88130E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30939E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26781E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91772E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.44068E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10985E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77364E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.19894E-02 -4.85521E+27  2.25652E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02789E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.59141E+18 0.00395  1.69864E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  8.47184E+19 0.00053  9.04273E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  4.65686E+18 0.00234  4.97048E-02 0.00223 ];
U238_FISS                 (idx, [1:   4]) = [  2.55145E+13 1.00000  2.71157E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  1.80236E+16 0.03984  1.92370E-04 0.03982 ];
PU240_FISS                (idx, [1:   4]) = [  5.71432E+14 0.20560  6.10410E-06 0.20561 ];
PU241_FISS                (idx, [1:   4]) = [  6.49218E+14 0.20772  6.91343E-06 0.20785 ];
TH232_CAPT                (idx, [1:   4]) = [  9.87538E+19 0.00050  6.94420E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23901E+19 0.00141  8.71261E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.62134E+18 0.00387  1.14008E-02 0.00383 ];
U238_CAPT                 (idx, [1:   4]) = [  1.55955E+14 0.40486  1.09789E-06 0.40487 ];
PU239_CAPT                (idx, [1:   4]) = [  8.82892E+15 0.05834  6.20935E-05 0.05831 ];
PU240_CAPT                (idx, [1:   4]) = [  1.65591E+15 0.12747  1.16427E-05 0.12733 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71580E+17 0.01252  1.20659E-03 0.01253 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002846 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13687E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002846 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5483456 5.48822E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3612574 3.61560E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906816 9.07557E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002846 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.07919E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36157E+20 3.7E-06  2.36157E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37032E+19 1.0E-07  9.37032E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42166E+20 0.00028  1.27536E+20 0.00019  1.46295E+19 0.00194 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35869E+20 0.00017  2.21240E+20 0.00011  1.46295E+19 0.00194 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59121E+20 0.00033  2.59121E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44031E+22 0.00023  3.26353E+22 0.00021  1.76778E+21 0.00185 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35175E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59387E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16510E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42390E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42390E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.73799E-01 0.05354 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24325E-01 0.01359 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.55800E-04 0.02678 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.98534E+03 0.02724 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09246E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.51272E-01 0.04244 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.92176E-01 0.04244 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52027E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99828E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11131E-01 0.00049  1.00903E-01 0.00048  3.40366E-04 0.00988 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11480E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11406E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11480E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00247E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92749E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92443E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47113E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47536E-02 0.00085 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83051E-01 0.00164 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83746E-01 0.00041 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.76788E-03 0.00585  2.88840E-04 0.02014  8.90646E-04 0.01055  6.84188E-04 0.01291  1.49667E-03 0.00851  3.36847E-04 0.01792  7.06950E-05 0.03601 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.90379E-01 0.01377  1.24760E-02 2.9E-05  3.22673E-02 0.00012  1.06593E-01 0.00056  3.00098E-01 0.00031  1.25416E+00 0.00069  6.47803E+00 0.02649 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35916E-03 0.00791  2.60798E-04 0.02972  7.87201E-04 0.01536  6.05877E-04 0.01826  1.33949E-03 0.01203  3.00587E-04 0.02647  6.52022E-05 0.05362 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.94757E-01 0.02064  1.24761E-02 3.8E-05  3.22664E-02 0.00015  1.06542E-01 0.00080  3.00203E-01 0.00044  1.25474E+00 0.00088  6.92642E+00 0.02233 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64176E-07 0.00117  8.63874E-07 0.00117  9.55163E-07 0.01522 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.87338E-07 0.00113  7.87063E-07 0.00113  8.70322E-07 0.01524 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.37024E-03 0.01012  2.59395E-04 0.03259  7.70174E-04 0.01828  5.90147E-04 0.02174  1.37461E-03 0.01483  3.07987E-04 0.03244  6.79312E-05 0.06532 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04028E-01 0.02461  1.24758E-02 5.3E-05  3.22662E-02 0.00021  1.06393E-01 0.00096  3.00102E-01 0.00050  1.25467E+00 0.00136  6.99157E+00 0.03197 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.66794E-07 0.01975  7.66649E-07 0.01975  7.89216E-07 0.05408 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.98549E-07 0.01974  6.98417E-07 0.01974  7.18485E-07 0.05400 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87187E-03 0.04648  2.43566E-04 0.14578  6.25213E-04 0.08829  5.13066E-04 0.10267  1.17257E-03 0.06647  2.54410E-04 0.14553  6.30385E-05 0.32205 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.70093E-01 0.11729  1.24754E-02 0.00016  3.22626E-02 0.00084  1.06275E-01 0.00298  2.98456E-01 0.00168  1.26071E+00 0.00424  6.00297E+00 0.12492 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.88592E-03 0.04587  2.43736E-04 0.14342  6.26399E-04 0.08419  5.15507E-04 0.10183  1.18693E-03 0.06578  2.48426E-04 0.14052  6.49129E-05 0.31644 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.71546E-01 0.11489  1.24756E-02 0.00016  3.22634E-02 0.00083  1.06271E-01 0.00295  2.98504E-01 0.00168  1.26046E+00 0.00420  5.99750E+00 0.12489 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.78073E+03 0.04284 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.57947E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.81652E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35811E-03 0.00567 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.91483E+03 0.00574 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76944E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71430E-05 0.00943  6.70645E-05 0.00946  1.84087E-06 0.41057 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.79141E-05 0.02726  5.79510E-05 0.02766  1.66082E-06 0.47465 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.53139E-04 0.02612  1.53115E-04 0.02614  1.61630E-04 0.40595 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67325E+01 0.01169 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49146E+01 0.00025  3.97542E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 16:22:37 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01223E+00  1.00063E+00  1.00044E+00  9.99874E-01  1.00144E+00  9.97508E-01  9.98987E-01  9.98409E-01  1.00623E+00  9.97287E-01  9.98963E-01  9.94975E-01  9.97552E-01  1.00224E+00  9.91604E-01  9.97542E-01  9.97125E-01  1.00697E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.09805E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.59020E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39730E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53492E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64125E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48935E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48030E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19418E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.74870E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333868 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33356E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33356E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.99595E+03 ;
RUNNING_TIME              (idx, 1)        =  5.21098E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.35807E+00  1.40600E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.18267E+02  5.65821E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.71067E-01  1.99167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.14626E+01  3.16037E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.17975E+02  8.02208E+02 ];
CPU_USAGE                 (idx, 1)        = 9.58735 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99068E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.54811E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.90023E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73762E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.46876E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01140E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38683E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85485E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56513E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.02840E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98917E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.85255E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.37691E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13548E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04352E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.82178E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38529E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35232E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57378E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92588E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73911E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27446E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96719E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.86705E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11548E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.76540E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.70946E-02 -5.98241E+27  2.26780E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02564E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.57762E+18 0.00396  1.68451E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  8.36345E+19 0.00052  8.92987E-01 0.00018 ];
U235_FISS                 (idx, [1:   4]) = [  5.34698E+18 0.00217  5.70903E-02 0.00209 ];
U238_FISS                 (idx, [1:   4]) = [  5.18267E+13 0.70592  5.51887E-07 0.70599 ];
PU239_FISS                (idx, [1:   4]) = [  4.54583E+16 0.02422  4.85619E-04 0.02430 ];
PU240_FISS                (idx, [1:   4]) = [  1.70990E+15 0.12269  1.82666E-05 0.12273 ];
PU241_FISS                (idx, [1:   4]) = [  1.62894E+15 0.12091  1.74244E-05 0.12088 ];
TH232_CAPT                (idx, [1:   4]) = [  9.75241E+19 0.00050  6.86661E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21969E+19 0.00139  8.58789E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.88300E+18 0.00372  1.32587E-02 0.00374 ];
U238_CAPT                 (idx, [1:   4]) = [  2.58182E+14 0.34274  1.82745E-06 0.34299 ];
PU239_CAPT                (idx, [1:   4]) = [  2.37170E+16 0.03327  1.66956E-04 0.03324 ];
PU240_CAPT                (idx, [1:   4]) = [  4.68522E+15 0.07340  3.29618E-05 0.07332 ];
PU241_CAPT                (idx, [1:   4]) = [  3.10765E+14 0.28332  2.18797E-06 0.28333 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71187E+17 0.01195  1.20526E-03 0.01194 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000668 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09173E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000668 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5481215 5.48694E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3614613 3.61827E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904840 9.05708E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000668 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.38419E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36145E+20 4.5E-06  2.36145E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36890E+19 1.1E-07  9.36890E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42002E+20 0.00029  1.27497E+20 0.00021  1.45053E+19 0.00193 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35691E+20 0.00018  2.21186E+20 0.00012  1.45053E+19 0.00193 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58847E+20 0.00035  2.58847E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43268E+22 0.00025  3.25751E+22 0.00022  1.75172E+21 0.00184 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.34448E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59136E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16091E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39605E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39605E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.88975E-01 0.05460 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.41393E-01 0.01279 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.50726E-04 0.02816 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.25411E+03 0.02595 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09432E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.62165E-01 0.04151 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.02236E-01 0.04151 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52052E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99858E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11722E-01 0.00044  1.00987E-01 0.00042  3.46548E-04 0.00872 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12300E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.12330E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12300E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00316E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90286E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90290E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50777E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50750E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.88693E-01 0.00147 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.88557E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.84215E-03 0.00541  2.87493E-04 0.02032  9.02572E-04 0.01068  6.93168E-04 0.01250  1.53601E-03 0.00837  3.48811E-04 0.01892  7.41013E-05 0.03804 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.94486E-01 0.01431  1.24351E-02 0.00334  3.22540E-02 0.00013  1.06546E-01 0.00055  3.00472E-01 0.00030  1.25762E+00 0.00077  6.18652E+00 0.02795 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43131E-03 0.00745  2.53560E-04 0.02845  8.13127E-04 0.01533  6.17100E-04 0.01722  1.37636E-03 0.01209  3.02129E-04 0.02725  6.90349E-05 0.05463 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.04358E-01 0.02185  1.24766E-02 3.7E-05  3.22535E-02 0.00018  1.06498E-01 0.00079  3.00393E-01 0.00042  1.25800E+00 0.00101  7.03270E+00 0.02193 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.55660E-07 0.00122  8.55081E-07 0.00122  1.02813E-06 0.02707 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.80091E-07 0.00118  7.79564E-07 0.00118  9.37090E-07 0.02701 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42349E-03 0.00879  2.54579E-04 0.03262  8.08536E-04 0.01918  6.24650E-04 0.02063  1.36212E-03 0.01420  3.13167E-04 0.03043  6.04355E-05 0.06856 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.80858E-01 0.02443  1.24759E-02 6.1E-05  3.22560E-02 0.00023  1.06612E-01 0.00095  3.00476E-01 0.00054  1.25894E+00 0.00145  6.73419E+00 0.03437 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.55840E-07 0.01957  7.55306E-07 0.01958  9.21554E-07 0.08446 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.89439E-07 0.01957  6.88949E-07 0.01957  8.41492E-07 0.08508 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.95520E-03 0.04341  2.41963E-04 0.12987  6.60530E-04 0.07557  4.94908E-04 0.09533  1.17707E-03 0.06600  2.96362E-04 0.14255  8.43717E-05 0.23737 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.03361E-01 0.09687  1.24778E-02 0.00011  3.22276E-02 0.00088  1.06647E-01 0.00330  3.00695E-01 0.00207  1.25463E+00 0.00387  6.25450E+00 0.09822 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97773E-03 0.04278  2.37614E-04 0.12418  6.72340E-04 0.07522  5.03135E-04 0.09035  1.17544E-03 0.06396  3.01621E-04 0.13882  8.75763E-05 0.22592 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.13175E-01 0.09375  1.24778E-02 0.00011  3.22259E-02 0.00087  1.06653E-01 0.00327  3.00678E-01 0.00206  1.25473E+00 0.00387  6.24241E+00 0.09821 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.92207E+03 0.03896 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.50470E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.75350E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43923E-03 0.00501 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.04435E+03 0.00503 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.74944E-09 0.00073 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72875E-05 0.00906  6.73381E-05 0.00908  9.87385E-07 0.46328 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.70770E-05 0.02734  5.71428E-05 0.02742  6.25658E-07 0.45989 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.49062E-04 0.02660  1.49034E-04 0.02673  1.56869E-04 0.47049 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65742E+01 0.01160 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48030E+01 0.00025  3.95134E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 17:20:27 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01063E+00  1.00627E+00  9.98482E-01  9.96184E-01  9.98663E-01  1.00172E+00  9.98193E-01  1.00065E+00  9.98320E-01  1.00277E+00  9.93866E-01  9.99761E-01  9.99624E-01  1.00014E+00  1.00029E+00  9.97375E-01  1.00088E+00  9.96184E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.10759E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58924E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39542E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53332E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64308E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47810E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46901E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17920E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.75142E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333833 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33369E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33369E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.54474E+03 ;
RUNNING_TIME              (idx, 1)        =  5.78942E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.50160E+00  1.43533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.75919E+02  5.76523E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.88483E-01  1.74167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36500E-02  1.36500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.45892E+01  3.12657E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.75853E+02  8.02826E+02 ];
CPU_USAGE                 (idx, 1)        = 9.57736 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99089E+00 1.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.53698E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.89972E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73823E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.08000E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99618E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37613E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86955E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56679E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.08676E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.16930E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.83097E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.53937E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21231E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20679E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20632E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37693E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34477E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56410E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.94940E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.12089E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26639E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99123E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.18773E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09401E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77036E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.22551E-02 -7.12184E+27  2.27919E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02404E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.57276E+18 0.00436  1.67888E-02 0.00427 ];
U233_FISS                 (idx, [1:   4]) = [  8.28006E+19 0.00055  8.83949E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.85874E+18 0.00201  6.25467E-02 0.00197 ];
PU239_FISS                (idx, [1:   4]) = [  8.34319E+16 0.01765  8.90734E-04 0.01766 ];
PU240_FISS                (idx, [1:   4]) = [  3.46384E+15 0.08529  3.70014E-05 0.08532 ];
PU241_FISS                (idx, [1:   4]) = [  5.16966E+15 0.07092  5.52239E-05 0.07098 ];
TH232_CAPT                (idx, [1:   4]) = [  9.66200E+19 0.00057  6.80072E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20994E+19 0.00148  8.51642E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.05898E+18 0.00357  1.44925E-02 0.00355 ];
U238_CAPT                 (idx, [1:   4]) = [  2.32896E+14 0.32887  1.64317E-06 0.32888 ];
PU239_CAPT                (idx, [1:   4]) = [  3.98755E+16 0.02648  2.80714E-04 0.02646 ];
PU240_CAPT                (idx, [1:   4]) = [  9.76611E+15 0.05159  6.87175E-05 0.05159 ];
PU241_CAPT                (idx, [1:   4]) = [  6.74210E+14 0.18775  4.73645E-06 0.18776 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77032E+17 0.01304  1.24612E-03 0.01304 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001063 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10690E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001063 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5479493 5.48521E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3613037 3.61652E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908533 9.09341E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001063 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.37836E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36156E+20 4.2E-06  2.36156E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36766E+19 9.7E-08  9.36766E+19 9.7E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.41990E+20 0.00031  1.27574E+20 0.00022  1.44168E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35667E+20 0.00019  2.21250E+20 0.00013  1.44168E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59012E+20 0.00036  2.59012E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43050E+22 0.00025  3.25640E+22 0.00023  1.74103E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35539E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59221E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15876E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.36899E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.36899E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.16526E-01 0.05971 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17926E-01 0.01357 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.57558E-04 0.02731 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.43937E+03 0.02677 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09068E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.08581E-01 0.04657 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.53348E-01 0.04657 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52097E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99885E+02 9.7E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11538E-01 0.00048  1.00952E-01 0.00046  3.46892E-04 0.00903 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12044E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11791E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12044E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00331E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88516E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88451E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53475E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53545E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.92264E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.92662E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85623E-03 0.00537  2.81613E-04 0.01934  9.02264E-04 0.01106  6.92502E-04 0.01254  1.55416E-03 0.00826  3.48663E-04 0.01872  7.70295E-05 0.03659 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.98321E-01 0.01387  1.24764E-02 4.2E-05  3.22451E-02 0.00014  1.06648E-01 0.00056  3.00593E-01 0.00029  1.25783E+00 0.00079  6.26936E+00 0.02721 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46168E-03 0.00773  2.51225E-04 0.02896  8.20141E-04 0.01500  6.17329E-04 0.01849  1.39017E-03 0.01177  3.11621E-04 0.02612  7.11975E-05 0.05727 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.98051E-01 0.02083  1.24761E-02 5.1E-05  3.22397E-02 0.00019  1.06541E-01 0.00077  3.00500E-01 0.00042  1.25694E+00 0.00103  6.90933E+00 0.02250 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.46400E-07 0.00121  8.46000E-07 0.00122  9.57420E-07 0.01596 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.71481E-07 0.00114  7.71116E-07 0.00115  8.72625E-07 0.01595 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42981E-03 0.00897  2.51458E-04 0.03334  8.10566E-04 0.01837  6.21145E-04 0.01994  1.38291E-03 0.01479  2.93773E-04 0.03016  6.99539E-05 0.06327 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.89036E-01 0.02524  1.24761E-02 4.8E-05  3.22399E-02 0.00024  1.06686E-01 0.00102  3.00502E-01 0.00051  1.25526E+00 0.00147  6.30351E+00 0.03530 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.45133E-07 0.01959  7.44848E-07 0.01959  8.03306E-07 0.04918 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.79489E-07 0.01959  6.79230E-07 0.01959  7.32349E-07 0.04912 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94321E-03 0.04527  2.13277E-04 0.13874  7.37952E-04 0.08189  5.34595E-04 0.09873  1.18120E-03 0.06811  2.09908E-04 0.12787  6.62785E-05 0.27355 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.46555E-01 0.07545  1.24782E-02 8.9E-05  3.22180E-02 0.00070  1.07092E-01 0.00365  3.00154E-01 0.00180  1.25546E+00 0.00452  6.23807E+00 0.08928 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.92969E-03 0.04443  2.10646E-04 0.13423  7.31086E-04 0.07949  5.29752E-04 0.09439  1.17179E-03 0.06638  2.17073E-04 0.12600  6.93450E-05 0.26891 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.52058E-01 0.07662  1.24782E-02 9.0E-05  3.22229E-02 0.00071  1.07060E-01 0.00357  3.00154E-01 0.00180  1.25581E+00 0.00442  6.22525E+00 0.08934 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96574E+03 0.04112 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.39314E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.65010E-07 0.00054 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42819E-03 0.00495 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.08544E+03 0.00505 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.74605E-09 0.00071 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67462E-05 0.01051  6.67625E-05 0.01057  1.48063E-06 0.39596 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.69203E-05 0.02929  5.69593E-05 0.02920  1.06225E-06 0.59269 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.55849E-04 0.02630  1.55754E-04 0.02641  1.92742E-04 0.37616 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63426E+01 0.01132 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46901E+01 0.00028  3.93325E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 18:19:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01205E+00  1.00239E+00  9.99143E-01  9.97487E-01  9.93048E-01  1.00023E+00  1.00669E+00  9.98570E-01  9.94126E-01  9.94886E-01  9.98222E-01  1.00225E+00  1.00494E+00  9.94459E-01  9.99559E-01  9.97257E-01  1.00525E+00  9.99447E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.12776E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58722E-01 1.4E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39346E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53181E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64563E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46960E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46056E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16915E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.76614E+00 0.00042  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334353 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33417E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33417E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.10130E+03 ;
RUNNING_TIME              (idx, 1)        =  6.37616E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.65342E+00  1.51817E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.34387E+02  5.84679E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.06450E-01  1.79667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36500E-02  1.36500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.77656E+01  3.17633E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.34484E+02  8.07950E+02 ];
CPU_USAGE                 (idx, 1)        = 9.56893 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99085E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.52763E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.92206E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74426E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.34233E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96280E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35629E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92524E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57479E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.60134E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.74835E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.10997E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03561E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44252E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70310E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34460E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37081E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33726E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55547E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99191E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27294E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26306E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04799E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.40908E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07281E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.79060E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.28127E-02 -1.16609E+28  2.32458E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02219E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.52962E+18 0.00410  1.63384E-02 0.00406 ];
U233_FISS                 (idx, [1:   4]) = [  8.07678E+19 0.00055  8.62721E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.83222E+18 0.00204  7.29771E-02 0.00195 ];
U238_FISS                 (idx, [1:   4]) = [  1.56620E+14 0.40484  1.66686E-06 0.40486 ];
PU239_FISS                (idx, [1:   4]) = [  2.86286E+17 0.00922  3.05787E-03 0.00919 ];
PU240_FISS                (idx, [1:   4]) = [  2.11124E+16 0.03458  2.25354E-04 0.03451 ];
PU241_FISS                (idx, [1:   4]) = [  3.03224E+16 0.02884  3.24041E-04 0.02890 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44567E+19 0.00051  6.61340E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17540E+19 0.00149  8.22958E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.41156E+18 0.00333  1.68848E-02 0.00331 ];
U238_CAPT                 (idx, [1:   4]) = [  1.08983E+15 0.15835  7.62895E-06 0.15834 ];
PU239_CAPT                (idx, [1:   4]) = [  1.38273E+17 0.01322  9.68228E-04 0.01324 ];
PU240_CAPT                (idx, [1:   4]) = [  5.45971E+16 0.02201  3.82175E-04 0.02196 ];
PU241_CAPT                (idx, [1:   4]) = [  6.43350E+15 0.06676  4.50670E-05 0.06681 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75264E+17 0.01168  1.22711E-03 0.01168 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002496 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08385E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002496 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5495161 5.49998E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3602375 3.60514E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904960 9.05725E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002496 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -5.34579E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36301E+20 4.7E-06  2.36301E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36375E+19 9.3E-08  9.36375E+19 9.3E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42789E+20 0.00030  1.28092E+20 0.00020  1.46975E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36427E+20 0.00018  2.21729E+20 0.00012  1.46975E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59687E+20 0.00033  2.59687E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44011E+22 0.00025  3.26299E+22 0.00021  1.77118E+21 0.00196 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35210E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59948E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15955E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26497E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26497E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.90322E-01 0.06158 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.13155E-01 0.01471 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.63472E-04 0.02693 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.04312E+03 0.02830 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09431E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.86739E-01 0.04855 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.33598E-01 0.04855 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52358E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99968E+02 9.3E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09632E-01 0.00045  1.00738E-01 0.00044  3.52443E-04 0.00918 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.10039E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09977E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.10039E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00068E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84183E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84396E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.60267E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59901E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.02491E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.02189E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93278E-03 0.00533  2.96063E-04 0.01829  9.07266E-04 0.01129  6.96645E-04 0.01203  1.57707E-03 0.00825  3.73233E-04 0.01704  8.25029E-05 0.03489 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.20792E-01 0.01413  1.24780E-02 5.9E-05  3.22245E-02 0.00014  1.06739E-01 0.00054  3.00951E-01 0.00031  1.26095E+00 0.00084  6.80522E+00 0.02332 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51095E-03 0.00744  2.69283E-04 0.02719  8.15520E-04 0.01637  6.20039E-04 0.01865  1.40370E-03 0.01224  3.28347E-04 0.02447  7.40610E-05 0.04891 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.20688E-01 0.02052  1.24776E-02 4.9E-05  3.22300E-02 0.00020  1.06717E-01 0.00080  3.00885E-01 0.00045  1.25967E+00 0.00106  7.32464E+00 0.01896 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.33651E-07 0.00129  8.33281E-07 0.00130  9.47132E-07 0.01678 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.58282E-07 0.00125  7.57945E-07 0.00126  8.61561E-07 0.01678 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48346E-03 0.00926  2.55445E-04 0.03312  8.12919E-04 0.01820  6.16671E-04 0.02127  1.39806E-03 0.01463  3.27168E-04 0.03056  7.31980E-05 0.06146 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.17873E-01 0.02441  1.24761E-02 7.1E-05  3.22146E-02 0.00024  1.06837E-01 0.00097  3.01047E-01 0.00051  1.26048E+00 0.00179  7.22001E+00 0.02821 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.35805E-07 0.01964  7.35589E-07 0.01963  7.97075E-07 0.05688 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.69476E-07 0.01962  6.69279E-07 0.01962  7.25171E-07 0.05691 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16235E-03 0.04277  1.94272E-04 0.14285  6.72034E-04 0.08396  5.26152E-04 0.09421  1.34454E-03 0.06107  3.70845E-04 0.11954  5.45095E-05 0.25396 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.00126E-01 0.11832  1.24725E-02 0.00021  3.22397E-02 0.00080  1.06403E-01 0.00298  3.00675E-01 0.00190  1.26009E+00 0.00451  7.14119E+00 0.08598 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.15582E-03 0.04297  1.92586E-04 0.14027  6.90225E-04 0.08293  5.25091E-04 0.09284  1.33060E-03 0.05963  3.68388E-04 0.11772  4.89346E-05 0.25505 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.80923E-01 0.11332  1.24721E-02 0.00022  3.22341E-02 0.00080  1.06424E-01 0.00299  3.00620E-01 0.00189  1.26057E+00 0.00438  7.14119E+00 0.08598 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.30008E+03 0.03807 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.26899E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.52131E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53170E-03 0.00551 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.27134E+03 0.00552 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.75877E-09 0.00074 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.86610E-05 0.00998  6.86494E-05 0.01000  1.54173E-06 0.42820 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.73786E-05 0.02958  5.73454E-05 0.02962  1.03803E-06 0.51911 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.64155E-04 0.02588  1.64201E-04 0.02587  1.62648E-04 0.40623 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63134E+01 0.01103 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46056E+01 0.00027  3.89372E+01 0.00037 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 19:18:04 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01279E+00  1.00417E+00  1.00064E+00  9.96424E-01  1.00006E+00  1.00121E+00  9.95861E-01  9.96807E-01  9.99677E-01  1.00184E+00  9.95351E-01  1.00109E+00  9.96865E-01  9.99726E-01  1.00426E+00  9.98443E-01  9.98252E-01  9.96537E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.18816E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.58118E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37867E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51866E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.66135E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46416E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45505E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20619E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.82815E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333639 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33368E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33368E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.66071E+03 ;
RUNNING_TIME              (idx, 1)        =  6.96549E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.80272E+00  1.49300E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.93111E+02  5.87243E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.26017E-01  1.95667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36500E-02  1.36500E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.09068E+01  3.14113E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.93446E+02  8.10720E+02 ];
CPU_USAGE                 (idx, 1)        = 9.56244 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99100E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.52050E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.91098E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73918E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.15290E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92204E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33600E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95493E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57175E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53593E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38927E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36124E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14123E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69228E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23727E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58073E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34472E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31615E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52732E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02229E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59747E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23229E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02867E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.14724E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01353E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.81682E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06409E-01 -2.34947E+28  2.44292E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01909E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.50552E+18 0.00419  1.60840E-02 0.00414 ];
U233_FISS                 (idx, [1:   4]) = [  7.93764E+19 0.00052  8.48024E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.13443E+18 0.00185  7.62229E-02 0.00182 ];
U238_FISS                 (idx, [1:   4]) = [  1.30159E+14 0.44425  1.39393E-06 0.44425 ];
PU239_FISS                (idx, [1:   4]) = [  5.98520E+17 0.00658  6.39456E-03 0.00658 ];
PU240_FISS                (idx, [1:   4]) = [  7.28273E+16 0.01836  7.77953E-04 0.01833 ];
PU241_FISS                (idx, [1:   4]) = [  1.22755E+17 0.01377  1.31116E-03 0.01370 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27698E+19 0.00054  6.46493E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15680E+19 0.00150  8.06165E-02 0.00147 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50478E+18 0.00314  1.74555E-02 0.00312 ];
U238_CAPT                 (idx, [1:   4]) = [  2.11187E+15 0.10862  1.47251E-05 0.10864 ];
PU239_CAPT                (idx, [1:   4]) = [  2.78857E+17 0.00913  1.94341E-03 0.00914 ];
PU240_CAPT                (idx, [1:   4]) = [  1.82550E+17 0.01220  1.27246E-03 0.01226 ];
PU241_CAPT                (idx, [1:   4]) = [  2.56083E+16 0.03168  1.78539E-04 0.03173 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81828E+17 0.01163  1.26726E-03 0.01165 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001031 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11221E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001031 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5501431 5.50725E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3588887 3.59234E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 910713 9.11537E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001031 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 6.89179E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36607E+20 4.9E-06  2.36607E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35966E+19 9.3E-08  9.35966E+19 9.3E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43441E+20 0.00028  1.28403E+20 0.00019  1.50380E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37037E+20 0.00017  2.21999E+20 0.00011  1.50380E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60561E+20 0.00032  2.60561E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47083E+22 0.00023  3.29108E+22 0.00020  1.79747E+21 0.00183 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37516E+19 0.00139 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60789E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16205E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.05590E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.05590E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.32249E-01 0.05448 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.22036E-01 0.01234 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.09467E-04 0.02451 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.90776E+03 0.02768 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08849E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.62872E-01 0.04120 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.02370E-01 0.04120 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52795E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00056E+02 9.3E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07950E-01 0.00043  1.00556E-01 0.00041  3.49149E-04 0.00894 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.08271E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08097E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.08271E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99388E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81311E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81466E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64939E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64654E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09416E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.09298E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93039E-03 0.00545  2.76592E-04 0.02042  8.99490E-04 0.01069  7.09706E-04 0.01311  1.60082E-03 0.00842  3.60956E-04 0.01698  8.28316E-05 0.03651 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.11504E-01 0.01388  1.24800E-02 7.7E-05  3.22006E-02 0.00016  1.06782E-01 0.00053  3.01141E-01 0.00028  1.25877E+00 0.00097  6.43979E+00 0.02566 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46480E-03 0.00754  2.48079E-04 0.02887  7.91904E-04 0.01501  6.13081E-04 0.01851  1.41776E-03 0.01262  3.19261E-04 0.02385  7.47149E-05 0.05280 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.16556E-01 0.02050  1.24794E-02 9.5E-05  3.21957E-02 0.00022  1.06719E-01 0.00074  3.01205E-01 0.00039  1.25920E+00 0.00136  7.00235E+00 0.02141 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.27803E-07 0.00126  8.27487E-07 0.00127  9.18483E-07 0.01633 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.51556E-07 0.00117  7.51270E-07 0.00118  8.33895E-07 0.01633 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45308E-03 0.00889  2.49093E-04 0.03482  8.00555E-04 0.01775  6.26510E-04 0.02176  1.38793E-03 0.01355  3.21472E-04 0.02992  6.75133E-05 0.06449 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.00748E-01 0.02286  1.24784E-02 9.5E-05  3.22148E-02 0.00029  1.06953E-01 0.00106  3.01151E-01 0.00050  1.26223E+00 0.00172  7.08450E+00 0.03024 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.31059E-07 0.01966  7.30620E-07 0.01966  8.50625E-07 0.05355 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.63802E-07 0.01965  6.63403E-07 0.01965  7.72478E-07 0.05354 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.25815E-03 0.04319  2.27688E-04 0.12669  7.65827E-04 0.08186  6.06160E-04 0.09614  1.32492E-03 0.06332  2.83764E-04 0.12136  4.97868E-05 0.30403 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.52144E-01 0.09374  1.24769E-02 0.00014  3.21597E-02 0.00094  1.06717E-01 0.00311  3.01305E-01 0.00192  1.26756E+00 0.00434  5.34045E+00 0.13321 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.24822E-03 0.04288  2.29605E-04 0.12571  7.58425E-04 0.07989  6.12199E-04 0.09664  1.31728E-03 0.06188  2.81808E-04 0.11945  4.89047E-05 0.29871 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.50769E-01 0.09277  1.24770E-02 0.00013  3.21602E-02 0.00093  1.06726E-01 0.00312  3.01157E-01 0.00189  1.26797E+00 0.00434  5.36094E+00 0.13233 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.49492E+03 0.03926 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.22814E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.47029E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.52667E-03 0.00572 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.28630E+03 0.00571 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.85458E-09 0.00089 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77431E-05 0.00687  6.77651E-05 0.00685  1.48534E-06 0.39180 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.71459E-05 0.02405  5.71055E-05 0.02405  1.72257E-06 0.56352 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.05768E-04 0.02325  2.05873E-04 0.02324  1.77849E-04 0.37565 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59623E+01 0.01076 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45505E+01 0.00027  3.86492E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 20:17:16 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00489E+00  1.00403E+00  1.00348E+00  9.94493E-01  1.00030E+00  9.97550E-01  9.95669E-01  9.97374E-01  1.00105E+00  9.99432E-01  9.91886E-01  1.00146E+00  9.97242E-01  1.00499E+00  9.98329E-01  9.96521E-01  1.00819E+00  1.00310E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.25254E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.57475E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36082E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50270E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.67824E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46278E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45355E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26113E+01 0.00033  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.89837E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334140 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33436E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33436E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.22204E+03 ;
RUNNING_TIME              (idx, 1)        =  7.55748E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.95262E+00  1.49900E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.52102E+02  5.89903E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.44100E-01  1.80833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.78500E-02  1.42000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.41266E+01  3.21973E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.52564E+02  8.11457E+02 ];
CPU_USAGE                 (idx, 1)        = 9.55614 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99090E+00 2.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.51359E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.90902E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73753E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.36193E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91174E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33110E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96315E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57047E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64299E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56581E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.46072E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16094E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76428E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39336E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93060E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33748E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31079E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51985E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04036E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00722E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22305E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01792E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.35491E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99750E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.83000E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61182E-01 -3.55885E+28  2.56386E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01393E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.50309E+18 0.00420  1.60444E-02 0.00412 ];
U233_FISS                 (idx, [1:   4]) = [  7.93071E+19 0.00058  8.46591E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.09113E+18 0.00185  7.56972E-02 0.00179 ];
U238_FISS                 (idx, [1:   4]) = [  1.82167E+14 0.42592  1.94760E-06 0.42574 ];
PU239_FISS                (idx, [1:   4]) = [  6.50091E+17 0.00618  6.93991E-03 0.00618 ];
PU240_FISS                (idx, [1:   4]) = [  8.51833E+16 0.01784  9.09420E-04 0.01787 ];
PU241_FISS                (idx, [1:   4]) = [  1.49305E+17 0.01235  1.59315E-03 0.01223 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21467E+19 0.00052  6.42290E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15540E+19 0.00160  8.05340E-02 0.00153 ];
U235_CAPT                 (idx, [1:   4]) = [  2.47153E+18 0.00321  1.72276E-02 0.00320 ];
U238_CAPT                 (idx, [1:   4]) = [  2.74028E+15 0.09359  1.91230E-05 0.09365 ];
PU239_CAPT                (idx, [1:   4]) = [  3.05518E+17 0.00958  2.12971E-03 0.00959 ];
PU240_CAPT                (idx, [1:   4]) = [  2.19996E+17 0.01045  1.53336E-03 0.01042 ];
PU241_CAPT                (idx, [1:   4]) = [  3.26884E+16 0.02778  2.27848E-04 0.02777 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81202E+17 0.01178  1.26314E-03 0.01178 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003084 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08458E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003084 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5492406 5.49683E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3586552 3.58922E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 924126 9.24792E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003084 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.40864E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36675E+20 5.4E-06  2.36675E+20 5.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35899E+19 1.0E-07  9.35899E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43412E+20 0.00029  1.28076E+20 0.00019  1.53364E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37002E+20 0.00017  2.21666E+20 0.00011  1.53364E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61000E+20 0.00034  2.61000E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49945E+22 0.00025  3.31881E+22 0.00022  1.80640E+21 0.00186 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41378E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61140E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16357E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87457E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87457E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.26415E-01 0.04315 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.01521E-01 0.01222 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.57349E-04 0.02185 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.85863E+03 0.02928 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07526E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.00467E-01 0.02892 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.26440E-01 0.02892 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52885E+00 5.3E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07680E-01 0.00046  1.00510E-01 0.00045  3.42179E-04 0.00898 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07302E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06833E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07302E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99814E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.80979E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81268E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65495E-02 0.00149 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64984E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10672E-01 0.00149 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10217E-01 0.00051 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.92033E-03 0.00541  2.69372E-04 0.02010  9.16123E-04 0.01139  7.02164E-04 0.01263  1.59136E-03 0.00827  3.61668E-04 0.01983  7.96440E-05 0.03779 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.04890E-01 0.01479  1.24778E-02 6.3E-05  3.21907E-02 0.00016  1.06857E-01 0.00059  3.01422E-01 0.00033  1.25365E+00 0.00349  6.27410E+00 0.02663 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.44290E-03 0.00852  2.34395E-04 0.02869  8.16294E-04 0.01572  6.08465E-04 0.01963  1.40064E-03 0.01205  3.11864E-04 0.02732  7.12424E-05 0.06011 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.03417E-01 0.02183  1.24781E-02 6.3E-05  3.21922E-02 0.00023  1.06872E-01 0.00084  3.01471E-01 0.00043  1.25774E+00 0.00147  6.86952E+00 0.02160 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.35024E-07 0.00142  8.34456E-07 0.00141  9.95180E-07 0.02004 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.57880E-07 0.00132  7.57365E-07 0.00132  9.03155E-07 0.01995 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39506E-03 0.00903  2.42145E-04 0.03174  8.01518E-04 0.01901  6.36330E-04 0.01933  1.34393E-03 0.01520  3.06569E-04 0.03261  6.45701E-05 0.06346 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.91200E-01 0.02495  1.24761E-02 6.5E-05  3.21844E-02 0.00029  1.06916E-01 0.00100  3.01616E-01 0.00058  1.25357E+00 0.00253  6.77198E+00 0.03303 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.33331E-07 0.01977  7.32936E-07 0.01977  8.29766E-07 0.05588 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.65881E-07 0.01976  6.65523E-07 0.01976  7.53677E-07 0.05595 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.13717E-03 0.04320  2.46877E-04 0.12664  7.23268E-04 0.08459  6.29519E-04 0.08535  1.21402E-03 0.07129  2.85113E-04 0.13806  3.83739E-05 0.38663 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.13594E-01 0.08198  1.24778E-02 0.00013  3.22206E-02 0.00081  1.06490E-01 0.00275  3.02748E-01 0.00215  1.25264E+00 0.00526  6.02860E+00 0.16010 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.10493E-03 0.04258  2.36492E-04 0.12719  7.13068E-04 0.08161  6.21222E-04 0.08537  1.21360E-03 0.06944  2.81240E-04 0.13677  3.93111E-05 0.38267 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.16545E-01 0.08069  1.24778E-02 0.00013  3.22203E-02 0.00081  1.06524E-01 0.00276  3.02643E-01 0.00212  1.25277E+00 0.00528  6.02860E+00 0.16010 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.32247E+03 0.03909 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.28042E-07 0.00085 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.51546E-07 0.00069 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.48850E-03 0.00524 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.21303E+03 0.00519 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99012E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.86926E-05 0.00575  6.87161E-05 0.00577  1.58732E-06 0.35176 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.12658E-05 0.01972  6.12480E-05 0.01978  1.71704E-06 0.49300 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.54734E-04 0.02043  2.54929E-04 0.02048  2.04834E-04 0.35005 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58058E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45355E+01 0.00028  3.86042E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha1' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 29 07:41:31 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 29 20:52:52 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651228891146 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01847E+00  9.99382E-01  9.96232E-01  1.00265E+00  9.98975E-01  1.00783E+00  9.94032E-01  9.91063E-01  9.94105E-01  9.97119E-01  1.00349E+00  9.98422E-01  9.99774E-01  1.00125E+00  9.99740E-01  1.00391E+00  9.96349E-01  9.97202E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  4.32233E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.56777E-01 1.6E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34699E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49073E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.69056E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47244E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46312E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.32256E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.97844E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334327 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33427E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33427E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  7.55691E+03 ;
RUNNING_TIME              (idx, 1)        =  7.91358E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.13595E+00  1.13595E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.10227E+00  1.49650E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.87499E+02  3.53975E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.62817E-01  1.87167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.78500E-02  1.42000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.63019E+01  2.17523E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.89220E+02  7.89220E+02 ];
CPU_USAGE                 (idx, 1)        = 9.54929 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98626E+00 3.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.50618E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22351.52;
MEMSIZE                   (idx, 1)        = 22051.86;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 499.43;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 299.66;

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

TOT_ACTIVITY              (idx, 1)        =  6.90542E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73689E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59287E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90529E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32727E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96592E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57014E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65635E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61277E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47186E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16170E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78523E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43933E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03032E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33557E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30942E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51789E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04176E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13328E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21766E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01452E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.39007E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98531E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.84478E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16503E-01 -4.78033E+28  2.68601E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01177E+00 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.50753E+18 0.00408  1.60999E-02 0.00406 ];
U233_FISS                 (idx, [1:   4]) = [  7.92671E+19 0.00054  8.46539E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.09556E+18 0.00182  7.57767E-02 0.00173 ];
U238_FISS                 (idx, [1:   4]) = [  1.81943E+14 0.37418  1.94045E-06 0.37417 ];
PU239_FISS                (idx, [1:   4]) = [  6.57479E+17 0.00618  7.02120E-03 0.00612 ];
PU240_FISS                (idx, [1:   4]) = [  8.83565E+16 0.01685  9.43543E-04 0.01682 ];
PU241_FISS                (idx, [1:   4]) = [  1.54669E+17 0.01222  1.65200E-03 0.01224 ];
TH232_CAPT                (idx, [1:   4]) = [  9.19048E+19 0.00060  6.39465E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15297E+19 0.00145  8.02254E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49210E+18 0.00322  1.73396E-02 0.00317 ];
U238_CAPT                 (idx, [1:   4]) = [  2.58669E+15 0.09926  1.80011E-05 0.09930 ];
PU239_CAPT                (idx, [1:   4]) = [  3.04196E+17 0.00924  2.11668E-03 0.00926 ];
PU240_CAPT                (idx, [1:   4]) = [  2.27565E+17 0.01009  1.58329E-03 0.01006 ];
PU241_CAPT                (idx, [1:   4]) = [  3.27950E+16 0.02556  2.28210E-04 0.02556 ];
SM149_CAPT                (idx, [1:   4]) = [  1.82141E+17 0.01190  1.26735E-03 0.01190 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002817 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06764E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002817 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5491736 5.49619E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3578134 3.58090E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 932947 9.33591E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002817 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.54600E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36685E+20 4.7E-06  2.36685E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35892E+19 1.0E-07  9.35892E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43711E+20 0.00029  1.27927E+20 0.00019  1.57845E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37300E+20 0.00017  2.21516E+20 0.00011  1.57845E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61493E+20 0.00035  2.61493E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53416E+22 0.00025  3.35014E+22 0.00023  1.84012E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.44135E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61714E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16826E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.71585E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.71585E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.06269E-01 0.04251 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07418E-01 0.01030 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.07334E-04 0.01898 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.18658E+03 0.03260 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06648E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 8.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.19026E-01 0.02710 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.42616E-01 0.02710 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52897E+00 4.7E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05565E-01 0.00045  1.00274E-01 0.00046  3.44281E-04 0.00901 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05347E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05161E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05347E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98596E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81515E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81433E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64603E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64711E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09884E-01 0.00144 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10328E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93427E-03 0.00562  2.86011E-04 0.02033  9.01152E-04 0.01182  7.00331E-04 0.01250  1.59713E-03 0.00785  3.64500E-04 0.01849  8.51469E-05 0.03783 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.17018E-01 0.01505  1.24795E-02 9.6E-05  3.22044E-02 0.00015  1.06866E-01 0.00057  3.01261E-01 0.00028  1.25728E+00 0.00104  6.31628E+00 0.02588 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47806E-03 0.00771  2.49326E-04 0.02988  7.94791E-04 0.01733  6.18468E-04 0.01632  1.42104E-03 0.01186  3.18745E-04 0.02510  7.56930E-05 0.05238 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.15759E-01 0.01998  1.24795E-02 0.00011  3.22021E-02 0.00022  1.06863E-01 0.00078  3.01313E-01 0.00039  1.25802E+00 0.00143  6.94562E+00 0.02119 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.47174E-07 0.00154  8.46647E-07 0.00155  1.01492E-06 0.03304 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.67122E-07 0.00147  7.66645E-07 0.00147  9.18989E-07 0.03292 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43378E-03 0.00909  2.52763E-04 0.03226  7.67612E-04 0.01845  6.03394E-04 0.02193  1.41422E-03 0.01462  3.14086E-04 0.02932  8.17038E-05 0.06358 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.33992E-01 0.02671  1.24799E-02 0.00017  3.21989E-02 0.00031  1.06903E-01 0.00107  3.01186E-01 0.00053  1.25479E+00 0.00211  7.03356E+00 0.02782 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.41209E-07 0.01974  7.40783E-07 0.01975  8.49378E-07 0.06422 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.71269E-07 0.01976  6.70883E-07 0.01976  7.69524E-07 0.06434 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.88843E-03 0.04094  2.25271E-04 0.14404  7.06634E-04 0.07966  4.47208E-04 0.10170  1.16717E-03 0.06093  2.59004E-04 0.13283  8.31454E-05 0.23555 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.36869E-01 0.11168  1.24717E-02 0.00023  3.21485E-02 0.00120  1.07220E-01 0.00367  3.01937E-01 0.00211  1.26218E+00 0.00571  6.62561E+00 0.07764 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87606E-03 0.04038  2.28212E-04 0.14838  6.95782E-04 0.07758  4.30485E-04 0.09856  1.17861E-03 0.05930  2.58079E-04 0.13154  8.48945E-05 0.23084 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.47644E-01 0.11150  1.24717E-02 0.00023  3.21475E-02 0.00120  1.07221E-01 0.00364  3.01967E-01 0.00211  1.26258E+00 0.00569  6.62882E+00 0.07758 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.91951E+03 0.03603 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.38265E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.59056E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39485E-03 0.00522 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.05076E+03 0.00530 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.14066E-09 0.00096 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.89753E-05 0.00548  6.89986E-05 0.00549  1.91563E-06 0.33845 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.47306E-05 0.01805  6.45736E-05 0.01814  2.84797E-06 0.43003 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.04766E-04 0.01754  3.04967E-04 0.01756  2.40923E-04 0.33823 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61241E+01 0.01137 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46312E+01 0.00027  3.86491E+01 0.00038 ];

