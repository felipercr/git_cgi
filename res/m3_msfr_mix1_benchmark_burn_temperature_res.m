
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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 15:27:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03190E+00  1.00648E+00  9.92430E-01  9.95252E-01  9.95722E-01  9.98519E-01  9.95237E-01  1.00361E+00  1.00708E+00  9.94860E-01  9.84607E-01  9.90843E-01  1.00992E+00  1.00055E+00  1.00337E+00  1.00392E+00  9.92073E-01  9.93625E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.31175E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66882E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40284E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54116E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62843E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45747E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44845E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.11373E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66485E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333540 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33390E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33390E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.69579E+02 ;
RUNNING_TIME              (idx, 1)        =  2.96800E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.82467E-01  2.82467E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.83027E+01  2.83027E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.89333E+00  1.84702E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.79839E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.08285 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97011E+00 0.00017 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.01407E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  2.80028E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.20219E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13449E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.80028E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.20219E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.69141E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.42870E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.69141E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.42870E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29717E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.13351E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.31722E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.15015E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.85701E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.55011E+18 0.00381  1.65290E-02 0.00376 ];
U233_FISS                 (idx, [1:   4]) = [  9.22289E+19 0.00048  9.83471E-01 6.3E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.35600E+19 0.00056  7.58801E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34018E+19 0.00129  1.08696E-01 0.00127 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001712 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14728E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001712 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5168028 5.17331E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3931071 3.93473E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 902613 9.03431E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001712 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.38304E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36450E+20 3.6E-06  2.36450E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37767E+19 1.2E-07  9.37767E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23262E+20 0.00032  1.10012E+20 0.00024  1.32493E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17038E+20 0.00018  2.03789E+20 0.00013  1.32493E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38338E+20 0.00036  2.38338E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.14446E+22 0.00027  2.98262E+22 0.00024  1.61845E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.15326E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38571E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06139E+22 0.00036 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.17956E-01 0.08123 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.78941E-01 0.00919 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.18520E-04 0.01620 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.18815E+04 0.02137 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09667E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.0E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.32861E-01 0.07133 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.93774E-01 0.07133 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52142E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.92012E-01 0.00040  1.09880E-01 0.00039  3.54365E-04 0.00908 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.92263E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.92117E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.92263E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09083E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02094E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01899E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33980E-02 0.00126 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34228E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.48533E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48483E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.24824E-03 0.00561  2.63589E-04 0.01943  8.15019E-04 0.01087  5.84892E-04 0.01326  1.26900E-03 0.00947  2.68628E-04 0.01955  4.71134E-05 0.04406 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40549E-01 0.01428  1.24760E-02 2.8E-05  3.23354E-02 9.3E-05  1.06052E-01 0.00059  2.97347E-01 0.00025  1.23592E+00 0.00022  5.29144E+00 0.03937 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19597E-03 0.00783  2.58218E-04 0.02805  7.97692E-04 0.01611  5.78850E-04 0.01909  1.24721E-03 0.01348  2.74158E-04 0.02712  3.98427E-05 0.06814 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.32163E-01 0.01991  1.24762E-02 3.6E-05  3.23312E-02 0.00012  1.06045E-01 0.00086  2.97326E-01 0.00038  1.23594E+00 0.00030  6.47313E+00 0.03052 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.19717E-07 0.00122  8.19473E-07 0.00123  8.94115E-07 0.01522 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.13127E-07 0.00114  8.12884E-07 0.00114  8.86960E-07 0.01522 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21076E-03 0.00931  2.69760E-04 0.03345  8.15174E-04 0.01807  5.71495E-04 0.02281  1.24020E-03 0.01565  2.71954E-04 0.02808  4.21754E-05 0.07292 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.34427E-01 0.02341  1.24760E-02 4.5E-05  3.23392E-02 0.00016  1.06180E-01 0.00106  2.97407E-01 0.00041  1.23581E+00 0.00038  6.41505E+00 0.04387 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.28381E-07 0.01959  7.28045E-07 0.01959  8.08383E-07 0.12373 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22735E-07 0.01958  7.22404E-07 0.01959  8.01435E-07 0.12311 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.77221E-03 0.04756  2.17415E-04 0.12912  7.10781E-04 0.08782  5.07256E-04 0.09317  1.06800E-03 0.06970  2.23684E-04 0.12564  4.50815E-05 0.34076 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.47298E-01 0.10580  1.24790E-02 2.8E-05  3.23301E-02 0.00052  1.05641E-01 0.00290  2.97308E-01 0.00151  1.23552E+00 0.00108  5.73702E+00 0.14438 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.75651E-03 0.04655  2.15106E-04 0.12483  6.95580E-04 0.08398  4.97821E-04 0.09220  1.07423E-03 0.06907  2.29960E-04 0.11917  4.38199E-05 0.34153 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.50184E-01 0.10647  1.24790E-02 3.1E-05  3.23315E-02 0.00052  1.05581E-01 0.00280  2.97381E-01 0.00151  1.23545E+00 0.00108  5.73702E+00 0.14438 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.81185E+03 0.04302 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.14109E-07 0.00067 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.07565E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.16941E-03 0.00585 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.89245E+03 0.00571 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.06684E-09 0.00127 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.28617E-05 0.00470  7.28396E-05 0.00471  3.68772E-06 0.26738 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.95138E-05 0.01675  7.95951E-05 0.01670  2.68584E-06 0.33959 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.12113E-04 0.01590  4.11951E-04 0.01601  4.70349E-04 0.26714 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73555E+01 0.01142 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44845E+01 0.00025  4.01641E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 16:12:04 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01751E+00  1.00510E+00  1.00176E+00  9.96508E-01  1.00005E+00  1.00382E+00  9.95832E-01  1.00301E+00  1.00027E+00  9.99123E-01  9.98330E-01  1.00032E+00  9.97419E-01  9.92672E-01  9.95650E-01  9.97884E-01  9.92251E-01  1.00248E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.30283E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66972E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40306E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54109E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62700E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46152E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45244E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.12125E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66083E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334011 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33382E+04 0.00051 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33382E+04 0.00051 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.91309E+02 ;
RUNNING_TIME              (idx, 1)        =  7.44588E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.99933E-01  1.17467E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.29415E+01  4.46388E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.74167E-02  1.74167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.23000E-02  1.23000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.67565E+00  2.78227E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.17180E+01  4.41406E+02 ];
CPU_USAGE                 (idx, 1)        = 9.28445 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97760E+00 0.00010 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.22611E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  5.79251E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60554E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65631E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.49639E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.05401E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26526E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46919E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.48813E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.66215E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.42496E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.83944E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06316E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17820E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44272E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35658E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17542E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55433E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.59953E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38167E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.40718E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51673E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51187E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.33839E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.22490E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08069E-04 -2.38612E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.95609E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.55743E+18 0.00415  1.65987E-02 0.00409 ];
U233_FISS                 (idx, [1:   4]) = [  9.22384E+19 0.00046  9.83101E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.86500E+14 0.24366  4.12052E-06 0.24366 ];
TH232_CAPT                (idx, [1:   4]) = [  9.45586E+19 0.00053  7.54090E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34054E+19 0.00127  1.06910E-01 0.00128 ];
U235_CAPT                 (idx, [1:   4]) = [  9.59356E+13 0.49751  7.62383E-07 0.49751 ];
SM149_CAPT                (idx, [1:   4]) = [  1.27601E+16 0.04215  1.01737E-04 0.04212 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001471 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15667E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001471 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5201301 5.20677E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3892075 3.89589E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908095 9.08908E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001471 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.63685E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36445E+20 3.6E-06  2.36445E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.1E-07  9.37764E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25382E+20 0.00030  1.11984E+20 0.00023  1.33975E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19158E+20 0.00017  2.05761E+20 0.00012  1.33975E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40830E+20 0.00036  2.40830E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.17851E+22 0.00025  3.01490E+22 0.00023  1.63615E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.18901E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41048E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07345E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51192E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51192E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.67633E-01 0.07450 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.93708E-01 0.00915 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.48440E-04 0.01721 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.17891E+04 0.01991 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09117E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 9.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.86468E-01 0.06394 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.42177E-01 0.06394 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52137E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.82350E-01 0.00038  1.08800E-01 0.00037  3.46437E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.82052E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.81830E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.82052E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08026E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01819E+00 0.00017 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01998E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34347E-02 0.00121 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34094E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49451E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49247E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.28965E-03 0.00552  2.72966E-04 0.02000  8.05147E-04 0.01126  5.94675E-04 0.01306  1.28867E-03 0.00874  2.75461E-04 0.01905  5.27346E-05 0.04304 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50159E-01 0.01442  1.24754E-02 2.9E-05  3.23312E-02 9.2E-05  1.06177E-01 0.00060  2.97324E-01 0.00026  1.23561E+00 0.00024  5.28829E+00 0.03743 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21079E-03 0.00786  2.67361E-04 0.02884  7.76140E-04 0.01779  5.79597E-04 0.01958  1.27140E-03 0.01339  2.67703E-04 0.02692  4.85943E-05 0.06553 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44623E-01 0.02096  1.24757E-02 3.7E-05  3.23242E-02 0.00011  1.06165E-01 0.00087  2.97303E-01 0.00038  1.23545E+00 0.00032  6.39819E+00 0.02893 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.27296E-07 0.00119  8.27030E-07 0.00120  9.11088E-07 0.01562 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.12667E-07 0.00116  8.12405E-07 0.00116  8.95024E-07 0.01563 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.15679E-03 0.00904  2.51249E-04 0.03050  7.69304E-04 0.01775  5.68750E-04 0.02180  1.24419E-03 0.01393  2.66982E-04 0.03022  5.63097E-05 0.06926 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.60620E-01 0.02492  1.24761E-02 4.7E-05  3.23267E-02 0.00014  1.06271E-01 0.00115  2.97332E-01 0.00043  1.23608E+00 0.00039  6.19087E+00 0.03943 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.40396E-07 0.01960  7.40123E-07 0.01961  7.96461E-07 0.05209 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.27526E-07 0.01960  7.27256E-07 0.01960  7.82996E-07 0.05214 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80700E-03 0.04461  2.21328E-04 0.14378  6.65080E-04 0.08666  4.84244E-04 0.09288  1.10500E-03 0.06756  2.68507E-04 0.12723  6.28453E-05 0.26151 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.50309E-01 0.12926  1.24750E-02 0.00016  3.23725E-02 0.00068  1.06306E-01 0.00358  2.96550E-01 0.00136  1.23700E+00 0.00097  7.51023E+00 0.09606 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85582E-03 0.04315  2.26775E-04 0.13981  6.76035E-04 0.08452  4.99303E-04 0.08898  1.11743E-03 0.06504  2.76498E-04 0.12311  5.97793E-05 0.25406 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.42311E-01 0.12871  1.24750E-02 0.00016  3.23717E-02 0.00068  1.06337E-01 0.00362  2.96635E-01 0.00139  1.23697E+00 0.00098  7.51023E+00 0.09606 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.80792E+03 0.04053 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.22849E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08289E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.14381E-03 0.00586 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82120E+03 0.00591 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99635E-09 0.00109 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.18824E-05 0.00530  7.18931E-05 0.00530  3.31901E-06 0.27164 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.92663E-05 0.01892  7.94292E-05 0.01909  3.26375E-06 0.39645 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.45146E-04 0.01660  3.44879E-04 0.01673  4.44652E-04 0.26334 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74530E+01 0.01156 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45244E+01 0.00028  4.01726E+01 0.00032 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 16:59:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01574E+00  1.00209E+00  9.98683E-01  1.00057E+00  1.00195E+00  1.00053E+00  1.00391E+00  1.00217E+00  9.97066E-01  1.00038E+00  9.91482E-01  9.97071E-01  1.00344E+00  9.89802E-01  9.97923E-01  9.97664E-01  1.00036E+00  9.99163E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.28941E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67106E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40273E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54021E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62422E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47275E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46371E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.14492E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.65792E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334572 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33391E+04 0.00055 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33391E+04 0.00055 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.13967E+03 ;
RUNNING_TIME              (idx, 1)        =  1.22119E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.28867E-01  1.28933E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.20446E+02  4.75041E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.63500E-02  1.89333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.69443E+00  3.01873E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.19149E+02  6.56589E+02 ];
CPU_USAGE                 (idx, 1)        = 9.33244 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97842E+00 0.00010 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.27625E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.41822E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65752E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.24012E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88591E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31804E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.50110E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49444E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.81368E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.41295E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.91428E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.10063E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89918E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60288E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49752E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34939E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33606E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54668E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.70260E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.18114E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.03291E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51937E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.33356E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.57307E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.30901E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.73371E-04 -1.26599E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.11792E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.57944E+18 0.00397  1.68423E-02 0.00391 ];
U233_FISS                 (idx, [1:   4]) = [  9.20700E+19 0.00055  9.81813E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.15853E+16 0.04601  1.23603E-04 0.04608 ];
TH232_CAPT                (idx, [1:   4]) = [  9.58766E+19 0.00049  7.48487E-01 0.00025 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34224E+19 0.00127  1.04787E-01 0.00124 ];
U235_CAPT                 (idx, [1:   4]) = [  3.45798E+15 0.08050  2.70084E-05 0.08062 ];
SM149_CAPT                (idx, [1:   4]) = [  5.94534E+16 0.01986  4.64238E-04 0.01989 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001717 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15889E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001717 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5252396 5.25768E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3845272 3.84904E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904049 9.04863E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001717 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.01700E-06 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36434E+20 3.7E-06  2.36434E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37759E+19 1.2E-07  9.37759E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28108E+20 0.00029  1.14542E+20 0.00021  1.35656E+19 0.00183 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.21884E+20 0.00017  2.08318E+20 0.00011  1.35656E+19 0.00183 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43634E+20 0.00036  2.43634E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.21994E+22 0.00025  3.05522E+22 0.00023  1.64714E+21 0.00181 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20462E+19 0.00156 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43930E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08869E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51428E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51428E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.71115E-01 0.05020 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17116E-01 0.01150 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.24601E-04 0.02176 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.02850E+04 0.02370 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09518E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.32968E-01 0.03907 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.66508E-01 0.03907 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52127E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.70289E-01 0.00041  1.07480E-01 0.00041  3.45334E-04 0.00928 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.70399E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.70487E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.70399E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06694E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02605E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02699E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33300E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33157E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49736E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49973E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.35611E-03 0.00576  2.81346E-04 0.02016  8.18048E-04 0.01130  6.02662E-04 0.01317  1.32784E-03 0.00909  2.75500E-04 0.01897  5.07200E-05 0.04341 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.40453E-01 0.01397  1.24756E-02 2.8E-05  3.23419E-02 0.00010  1.06198E-01 0.00059  2.97491E-01 0.00026  1.23562E+00 0.00024  5.07107E+00 0.03831 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23706E-03 0.00791  2.68813E-04 0.02805  8.12593E-04 0.01660  5.81743E-04 0.02049  1.26560E-03 0.01288  2.59288E-04 0.02808  4.90284E-05 0.07119 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.41044E-01 0.02394  1.24752E-02 4.1E-05  3.23358E-02 0.00013  1.06214E-01 0.00084  2.97529E-01 0.00039  1.23565E+00 0.00033  6.22267E+00 0.02978 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.41658E-07 0.00119  8.41399E-07 0.00120  9.22466E-07 0.01785 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16608E-07 0.00112  8.16357E-07 0.00112  8.95029E-07 0.01781 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19823E-03 0.00941  2.66225E-04 0.03132  7.87805E-04 0.01863  5.63186E-04 0.02108  1.26390E-03 0.01532  2.66072E-04 0.03352  5.10388E-05 0.07265 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.47764E-01 0.02441  1.24757E-02 5.4E-05  3.23406E-02 0.00017  1.05990E-01 0.00097  2.97434E-01 0.00042  1.23560E+00 0.00042  6.26241E+00 0.04150 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.53260E-07 0.01971  7.53028E-07 0.01971  7.91473E-07 0.05113 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.31234E-07 0.01970  7.31009E-07 0.01970  7.68085E-07 0.05103 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87332E-03 0.04438  2.26409E-04 0.14729  6.81092E-04 0.08246  5.24013E-04 0.09240  1.16620E-03 0.06400  2.27085E-04 0.14208  4.85166E-05 0.31697 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.57334E-01 0.09233  1.24768E-02 0.00012  3.23334E-02 0.00060  1.05576E-01 0.00274  2.96795E-01 0.00136  1.23525E+00 0.00128  6.59154E+00 0.11747 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.85990E-03 0.04335  2.23163E-04 0.14545  6.79894E-04 0.07907  5.27828E-04 0.09132  1.14802E-03 0.06322  2.31378E-04 0.13861  4.96135E-05 0.32422 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.50147E-01 0.08901  1.24769E-02 0.00012  3.23335E-02 0.00060  1.05584E-01 0.00273  2.96841E-01 0.00135  1.23527E+00 0.00128  6.59154E+00 0.11747 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.82627E+03 0.04004 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.37244E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12325E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.20376E-03 0.00509 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82779E+03 0.00525 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.84852E-09 0.00090 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.90487E-05 0.00690  6.90537E-05 0.00692  1.24520E-06 0.45690 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.63663E-05 0.02299  6.64145E-05 0.02303  1.04570E-06 0.54276 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.22582E-04 0.02158  2.22831E-04 0.02158  1.53934E-04 0.44729 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73077E+01 0.01178 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46371E+01 0.00026  4.03286E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 17:47:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01814E+00  9.98431E-01  9.93373E-01  1.00337E+00  9.96853E-01  9.95863E-01  1.00467E+00  1.00400E+00  1.00252E+00  9.98769E-01  9.91585E-01  9.98323E-01  9.95853E-01  1.00126E+00  9.96887E-01  9.98989E-01  9.99170E-01  1.00194E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.27714E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67229E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40436E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54144E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62002E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47918E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47013E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.15227E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.65093E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333799 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33403E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33403E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.59358E+03 ;
RUNNING_TIME              (idx, 1)        =  1.70363E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.55533E-01  1.26667E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.68533E+02  4.80878E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.45667E-02  1.82167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.07427E+01  3.04827E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.67357E+02  6.91496E+02 ];
CPU_USAGE                 (idx, 1)        = 9.35402 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97932E+00 9.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.29878E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.53628E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67384E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.15521E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92255E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34273E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58219E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50798E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.86902E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.57532E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.06878E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.29139E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79915E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74616E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48902E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36788E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35026E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56579E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39982E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42650E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10680E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54962E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.16513E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.73414E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.36917E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12004E-03 -2.47302E+26  2.21045E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.24370E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.58712E+18 0.00391  1.69218E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  9.19494E+19 0.00054  9.80324E-01 7.4E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.14439E+16 0.02528  4.41841E-04 0.02529 ];
TH232_CAPT                (idx, [1:   4]) = [  9.68107E+19 0.00052  7.45446E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34265E+19 0.00137  1.03386E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  1.48224E+16 0.04274  1.14022E-04 0.04265 ];
SM149_CAPT                (idx, [1:   4]) = [  9.55677E+16 0.01677  7.35925E-04 0.01680 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002077 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15277E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002077 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5281869 5.28705E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3814906 3.81841E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905302 9.06067E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002077 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.84288E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36432E+20 3.7E-06  2.36432E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.1E-07  9.37752E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29766E+20 0.00030  1.16115E+20 0.00023  1.36502E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23541E+20 0.00017  2.09891E+20 0.00013  1.36502E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.45639E+20 0.00036  2.45639E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24805E+22 0.00026  3.08212E+22 0.00023  1.65926E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.22574E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.45798E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09923E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51428E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51428E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.96893E-01 0.05207 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.15145E-01 0.01394 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.83512E-04 0.02544 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.87908E+03 0.02585 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09396E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.30157E-01 0.03877 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.64075E-01 0.03877 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52126E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.62684E-01 0.00044  1.06622E-01 0.00042  3.45681E-04 0.00868 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.63026E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.62554E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.63026E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05903E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02473E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02538E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33471E-02 0.00124 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33371E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51092E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51435E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.40314E-03 0.00569  2.80946E-04 0.01949  8.41205E-04 0.01177  6.28193E-04 0.01310  1.31376E-03 0.00894  2.79844E-04 0.01825  5.91947E-05 0.04153 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.59598E-01 0.01557  1.24754E-02 2.9E-05  3.23334E-02 8.9E-05  1.05982E-01 0.00058  2.97449E-01 0.00025  1.23586E+00 0.00024  5.69028E+00 0.03340 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.26176E-03 0.00829  2.71763E-04 0.03139  7.98277E-04 0.01743  5.96292E-04 0.02063  1.26890E-03 0.01283  2.70427E-04 0.02705  5.61040E-05 0.05782 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.64508E-01 0.02300  1.24755E-02 3.5E-05  3.23297E-02 0.00012  1.05985E-01 0.00078  2.97553E-01 0.00037  1.23552E+00 0.00034  6.54221E+00 0.02756 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.47733E-07 0.00134  8.47387E-07 0.00134  9.52475E-07 0.01591 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16046E-07 0.00125  8.15712E-07 0.00125  9.16988E-07 0.01594 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23255E-03 0.00886  2.64028E-04 0.03413  8.21741E-04 0.01974  5.90217E-04 0.02064  1.23046E-03 0.01482  2.70599E-04 0.03155  5.55060E-05 0.06656 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.55407E-01 0.02443  1.24752E-02 4.9E-05  3.23340E-02 0.00016  1.05854E-01 0.00088  2.97424E-01 0.00045  1.23548E+00 0.00044  6.32224E+00 0.04031 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.55405E-07 0.01983  7.54931E-07 0.01983  8.85521E-07 0.05533 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.27337E-07 0.01982  7.26880E-07 0.01982  8.52216E-07 0.05519 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89499E-03 0.04243  2.86055E-04 0.12495  7.26706E-04 0.07770  5.18990E-04 0.09097  1.08240E-03 0.06634  2.30935E-04 0.13269  4.99043E-05 0.31370 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.56773E-01 0.13413  1.24762E-02 0.00012  3.23538E-02 0.00057  1.06320E-01 0.00343  2.97374E-01 0.00169  1.23479E+00 0.00118  6.95210E+00 0.11692 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90057E-03 0.04173  2.81818E-04 0.12389  7.37157E-04 0.07590  5.19576E-04 0.09067  1.08072E-03 0.06491  2.29739E-04 0.13184  5.15610E-05 0.29168 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.60966E-01 0.12841  1.24764E-02 0.00012  3.23567E-02 0.00057  1.06339E-01 0.00343  2.97392E-01 0.00169  1.23496E+00 0.00116  6.95541E+00 0.11675 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.85058E+03 0.03755 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.43915E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12373E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32528E-03 0.00692 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.94149E+03 0.00702 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78616E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76832E-05 0.00707  6.76549E-05 0.00707  1.09932E-06 0.51383 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.43180E-05 0.02780  6.44097E-05 0.02782  4.01192E-07 0.51084 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.80290E-04 0.02500  1.80504E-04 0.02496  1.17635E-04 0.50159 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73420E+01 0.01117 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47013E+01 0.00025  4.03884E+01 0.00033 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 18:37:57 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00942E+00  9.98301E-01  9.91994E-01  1.00231E+00  1.00078E+00  9.98732E-01  9.98786E-01  1.00825E+00  1.00737E+00  1.00432E+00  9.93396E-01  9.93009E-01  1.00003E+00  9.96684E-01  1.00145E+00  1.00219E+00  9.92205E-01  1.00077E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.27075E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67293E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40550E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54231E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61481E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49739E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48834E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18236E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.65339E+00 0.00043  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333724 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33361E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33361E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.05955E+03 ;
RUNNING_TIME              (idx, 1)        =  2.20334E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.85500E-01  1.29967E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.18342E+02  4.98087E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.32333E-02  1.86667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.43528E+01  3.61003E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.16764E+02  6.99129E+02 ];
CPU_USAGE                 (idx, 1)        = 9.34740 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97931E+00 8.8E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.29113E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.56548E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67012E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.51139E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.94155E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35434E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.59214E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50287E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00513E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.69790E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.33665E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.37811E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.70898E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85997E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93941E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33731E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32693E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53353E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35753E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81734E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10623E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52884E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51309E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.77935E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.45416E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.26273E-03 -4.99604E+26  2.21297E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.43919E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.59963E+18 0.00396  1.70527E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  9.16321E+19 0.00051  9.76883E-01 8.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.56820E+17 0.01244  1.67190E-03 0.01244 ];
TH232_CAPT                (idx, [1:   4]) = [  9.80804E+19 0.00053  7.40561E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33969E+19 0.00139  1.01156E-01 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  5.69481E+16 0.02075  4.29953E-04 0.02073 ];
SM149_CAPT                (idx, [1:   4]) = [  1.36727E+17 0.01357  1.03241E-03 0.01357 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000833 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16572E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000833 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5324241 5.33023E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3771119 3.77511E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905473 9.06317E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000833 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 6.22123E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36419E+20 3.6E-06  2.36419E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37730E+19 1.1E-07  9.37730E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32493E+20 0.00030  1.18582E+20 0.00022  1.39115E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26266E+20 0.00017  2.12355E+20 0.00012  1.39115E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48472E+20 0.00035  2.48472E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29352E+22 0.00025  3.12443E+22 0.00023  1.69088E+21 0.00167 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.25201E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.48786E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11647E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51144E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51144E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.75819E-01 0.05409 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07195E-01 0.01582 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.58097E-04 0.02636 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.84982E+03 0.02551 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09371E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.01115E-01 0.04059 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.37472E-01 0.04059 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52118E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.51677E-01 0.00043  1.05407E-01 0.00042  3.43068E-04 0.00924 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.51400E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.51526E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.51400E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04621E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02688E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02467E+00 0.00012 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33189E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33464E-02 0.00087 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52743E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53532E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44475E-03 0.00579  2.82108E-04 0.01681  8.43060E-04 0.01204  6.39273E-04 0.01311  1.33701E-03 0.00930  2.90224E-04 0.01827  5.30787E-05 0.04297 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47026E-01 0.01467  1.24761E-02 2.5E-05  3.23296E-02 9.9E-05  1.06076E-01 0.00058  2.97619E-01 0.00030  1.23686E+00 0.00026  5.17646E+00 0.03808 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27356E-03 0.00832  2.56897E-04 0.02784  8.15202E-04 0.01784  6.16844E-04 0.01838  1.25809E-03 0.01319  2.75932E-04 0.02786  5.05967E-05 0.07158 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.51308E-01 0.02504  1.24756E-02 3.7E-05  3.23266E-02 0.00013  1.06063E-01 0.00082  2.97722E-01 0.00040  1.23678E+00 0.00036  6.34333E+00 0.02907 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.58708E-07 0.00120  8.58322E-07 0.00121  9.78333E-07 0.01621 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17173E-07 0.00115  8.16806E-07 0.00115  9.30907E-07 0.01619 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24250E-03 0.00944  2.57891E-04 0.03044  7.86661E-04 0.01949  6.10254E-04 0.02129  1.24762E-03 0.01475  2.82504E-04 0.02914  5.75613E-05 0.06660 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.65758E-01 0.02443  1.24766E-02 4.0E-05  3.23403E-02 0.00017  1.06000E-01 0.00098  2.97708E-01 0.00049  1.23744E+00 0.00055  6.35708E+00 0.03914 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.59336E-07 0.01963  7.59189E-07 0.01963  8.27532E-07 0.05238 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22659E-07 0.01963  7.22519E-07 0.01962  7.87426E-07 0.05244 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.89789E-03 0.04252  2.49421E-04 0.13296  7.15984E-04 0.08051  5.22217E-04 0.08805  1.09193E-03 0.06419  2.84700E-04 0.12831  3.36420E-05 0.25872 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.37939E-01 0.06881  1.24766E-02 0.00012  3.23205E-02 0.00052  1.06551E-01 0.00366  2.98471E-01 0.00189  1.23872E+00 0.00129  6.42536E+00 0.12268 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87841E-03 0.04196  2.47701E-04 0.13154  7.13108E-04 0.07815  5.19792E-04 0.08594  1.08828E-03 0.06317  2.77116E-04 0.12692  3.24121E-05 0.26061 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.33522E-01 0.06586  1.24769E-02 0.00011  3.23235E-02 0.00053  1.06568E-01 0.00367  2.98369E-01 0.00185  1.23845E+00 0.00118  6.44245E+00 0.12188 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.83772E+03 0.03827 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.53193E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11918E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22990E-03 0.00657 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.78680E+03 0.00669 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77773E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72645E-05 0.00908  6.72125E-05 0.00914  1.41304E-06 0.41173 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.09791E-05 0.02945  6.13040E-05 0.03000  1.15111E-06 0.51937 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.57570E-04 0.02506  1.57513E-04 0.02514  1.83436E-04 0.40727 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73561E+01 0.01109 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48834E+01 0.00025  4.04983E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 19:28:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01667E+00  9.99590E-01  9.95602E-01  9.97400E-01  9.89899E-01  1.00399E+00  9.99286E-01  1.00189E+00  9.99188E-01  9.97444E-01  9.94725E-01  1.00116E+00  1.00294E+00  9.97733E-01  1.00408E+00  9.93696E-01  1.00582E+00  9.98899E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.26906E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67309E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40499E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54179E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61233E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50965E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50064E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20664E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.65562E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333868 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33386E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33386E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.51280E+03 ;
RUNNING_TIME              (idx, 1)        =  2.70539E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.13950E-01  1.28450E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.68383E+02  5.00410E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.15500E-02  1.83167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.96416E+01  5.28873E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.65290E+02  7.14889E+02 ];
CPU_USAGE                 (idx, 1)        = 9.28812 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98100E+00 8.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.22476E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.62746E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67885E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.15483E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97187E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37418E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62346E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50928E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10177E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.80575E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.73801E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.51896E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.26488E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.95356E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35634E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.34642E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33364E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54256E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.03361E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17750E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14097E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55841E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.34632E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.86917E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.52304E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.39016E-03 -7.48537E+26  2.21546E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.62714E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.61240E+18 0.00406  1.71909E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  9.12430E+19 0.00051  9.72825E-01 8.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.32821E+17 0.00871  3.54857E-03 0.00871 ];
TH232_CAPT                (idx, [1:   4]) = [  9.92404E+19 0.00053  7.37063E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33197E+19 0.00130  9.89291E-02 0.00131 ];
U235_CAPT                 (idx, [1:   4]) = [  1.17917E+17 0.01449  8.75874E-04 0.01452 ];
SM149_CAPT                (idx, [1:   4]) = [  1.50700E+17 0.01252  1.11938E-03 0.01255 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001570 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14811E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001570 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5363727 5.36925E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3736567 3.74021E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 901276 9.02023E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001570 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.62633E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36406E+20 3.5E-06  2.36406E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37701E+19 1.1E-07  9.37701E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34661E+20 0.00031  1.20572E+20 0.00022  1.40884E+19 0.00191 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.28431E+20 0.00018  2.14342E+20 0.00012  1.40884E+19 0.00191 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.50768E+20 0.00033  2.50768E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.33069E+22 0.00024  3.15960E+22 0.00022  1.71089E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.26204E+19 0.00141 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.51051E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12984E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50738E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50738E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.81951E-01 0.05403 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.23529E-01 0.01451 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.55855E-04 0.02602 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.64022E+03 0.02612 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09800E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.97801E-01 0.04028 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.34923E-01 0.04028 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52112E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.42936E-01 0.00045  1.04440E-01 0.00044  3.30953E-04 0.00918 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.42761E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.42759E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.42761E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03624E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02618E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02248E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33280E-02 0.00128 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33758E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.55467E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.55911E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.46731E-03 0.00599  2.79013E-04 0.02005  8.40021E-04 0.01140  6.33556E-04 0.01305  1.36702E-03 0.00914  2.89513E-04 0.01927  5.81874E-05 0.04404 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.56459E-01 0.01505  1.24755E-02 2.9E-05  3.23240E-02 9.3E-05  1.06153E-01 0.00061  2.97750E-01 0.00026  1.23764E+00 0.00035  5.42730E+00 0.03636 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.19749E-03 0.00864  2.51217E-04 0.02863  7.69887E-04 0.01643  5.82998E-04 0.01882  1.27338E-03 0.01354  2.63370E-04 0.02972  5.66371E-05 0.06269 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.62624E-01 0.02185  1.24752E-02 4.0E-05  3.23206E-02 0.00013  1.06149E-01 0.00080  2.97626E-01 0.00038  1.23786E+00 0.00044  6.50927E+00 0.02835 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.69276E-07 0.00123  8.69091E-07 0.00123  9.26986E-07 0.01686 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.19627E-07 0.00116  8.19452E-07 0.00116  8.74147E-07 0.01690 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.16228E-03 0.00931  2.49913E-04 0.03172  7.65725E-04 0.01818  5.68592E-04 0.02144  1.26495E-03 0.01440  2.61873E-04 0.03212  5.12243E-05 0.07359 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.54339E-01 0.02588  1.24749E-02 6.0E-05  3.23180E-02 0.00015  1.06086E-01 0.00114  2.97755E-01 0.00044  1.23691E+00 0.00060  6.42004E+00 0.04142 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.70471E-07 0.01984  7.70000E-07 0.01984  8.82383E-07 0.07070 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.26727E-07 0.01984  7.26283E-07 0.01985  8.32204E-07 0.07065 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94981E-03 0.04408  2.63232E-04 0.11819  6.73565E-04 0.08351  4.67185E-04 0.09691  1.22980E-03 0.06405  2.37574E-04 0.12345  7.84532E-05 0.25637 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.34832E-01 0.11465  1.24743E-02 0.00016  3.23024E-02 0.00052  1.06025E-01 0.00322  2.97033E-01 0.00147  1.23524E+00 0.00113  6.50510E+00 0.10310 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.94048E-03 0.04390  2.59834E-04 0.11392  6.78500E-04 0.08199  4.74959E-04 0.09534  1.20932E-03 0.06503  2.43564E-04 0.12183  7.43121E-05 0.24747 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.30076E-01 0.11325  1.24741E-02 0.00016  3.23029E-02 0.00052  1.06020E-01 0.00322  2.96998E-01 0.00146  1.23523E+00 0.00113  6.52008E+00 0.10272 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.84529E+03 0.03939 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.63537E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.14200E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18360E-03 0.00651 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.68751E+03 0.00657 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78013E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76855E-05 0.00898  6.77070E-05 0.00904  4.49141E-07 0.70730 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.97401E-05 0.02932  5.97976E-05 0.02926  4.28417E-08 0.84808 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52593E-04 0.02516  1.52924E-04 0.02520  5.45058E-05 0.70592 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71940E+01 0.01167 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50064E+01 0.00026  4.05607E+01 0.00034 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 20:19:22 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01438E+00  9.99612E-01  1.00662E+00  9.95133E-01  1.00445E+00  1.00150E+00  9.95432E-01  9.99303E-01  1.00515E+00  9.95000E-01  1.00134E+00  9.93070E-01  1.00032E+00  9.97725E-01  9.97044E-01  1.00147E+00  9.92178E-01  1.00028E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.26952E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67305E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40083E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53774E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61136E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51582E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50673E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.23234E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66059E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333888 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33397E+04 0.00067 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33397E+04 0.00067 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.97474E+03 ;
RUNNING_TIME              (idx, 1)        =  3.21755E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.04753E+00  1.33583E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.19427E+02  5.10437E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.10767E-01  1.92167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.50827E+01  5.44098E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.16355E+02  7.17905E+02 ];
CPU_USAGE                 (idx, 1)        = 9.24538 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98274E+00 7.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.17692E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.65590E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67806E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.12931E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99506E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38758E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62833E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50680E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27337E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.94093E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.73289E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.65489E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.95499E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.07453E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07935E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32350E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31525E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51762E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.85927E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.79577E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14669E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.57123E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.93204E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.91107E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.62255E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43284E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.71821E-03 -1.26256E+27  2.22060E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.89081E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.61969E+18 0.00403  1.72782E-02 0.00394 ];
U233_FISS                 (idx, [1:   4]) = [  9.03425E+19 0.00054  9.63802E-01 9.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  8.16591E+17 0.00553  8.71204E-03 0.00553 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00567E+20 0.00051  7.31282E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32109E+19 0.00128  9.60667E-02 0.00127 ];
U235_CAPT                 (idx, [1:   4]) = [  2.89194E+17 0.00950  2.10319E-03 0.00954 ];
PU239_CAPT                (idx, [1:   4]) = [  2.57118E+13 1.00000  1.85340E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.64154E+17 0.01230  1.19380E-03 0.01232 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001913 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14459E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001913 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5406995 5.41243E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3685806 3.68916E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 909112 9.09857E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001913 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 5.02914E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36375E+20 3.8E-06  2.36375E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37628E+19 1.1E-07  9.37628E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37530E+20 0.00029  1.23240E+20 0.00022  1.42898E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31292E+20 0.00017  2.17003E+20 0.00013  1.42898E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.54085E+20 0.00035  2.54085E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.38001E+22 0.00025  3.20746E+22 0.00023  1.72550E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.31188E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.54411E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14633E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49912E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49912E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.78756E-01 0.05622 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.14416E-01 0.01596 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.51197E-04 0.02567 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.41193E+03 0.02685 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09016E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.61740E-01 0.04275 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.01515E-01 0.04275 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52099E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.29781E-01 0.00048  1.02999E-01 0.00044  3.35663E-04 0.00925 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.30184E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.30331E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.30184E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02328E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01579E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01414E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34677E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34881E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.59673E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.60496E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.57761E-03 0.00523  2.97962E-04 0.01953  8.65200E-04 0.01130  6.48156E-04 0.01257  1.41358E-03 0.00821  2.98033E-04 0.02024  5.46794E-05 0.04510 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.48781E-01 0.01562  1.24754E-02 2.9E-05  3.23188E-02 0.00010  1.06280E-01 0.00061  2.98155E-01 0.00029  1.23952E+00 0.00040  5.08396E+00 0.04085 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.27452E-03 0.00797  2.70549E-04 0.02865  7.95791E-04 0.01743  5.92783E-04 0.01701  1.29436E-03 0.01323  2.73870E-04 0.02674  4.71618E-05 0.06107 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.44982E-01 0.02040  1.24753E-02 4.3E-05  3.23217E-02 0.00015  1.06197E-01 0.00077  2.98263E-01 0.00040  1.23876E+00 0.00045  6.52791E+00 0.02883 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.79823E-07 0.00120  8.79378E-07 0.00120  1.02559E-06 0.02949 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17987E-07 0.00110  8.17573E-07 0.00110  9.53517E-07 0.02955 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.24879E-03 0.00952  2.68189E-04 0.03197  7.91850E-04 0.01871  5.91133E-04 0.02091  1.27171E-03 0.01484  2.71235E-04 0.03035  5.46748E-05 0.07046 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.59441E-01 0.02640  1.24752E-02 6.0E-05  3.23170E-02 0.00018  1.06254E-01 0.00110  2.98089E-01 0.00048  1.23935E+00 0.00072  6.33053E+00 0.04005 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.79195E-07 0.01964  7.78697E-07 0.01964  9.05453E-07 0.05731 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.24731E-07 0.01965  7.24267E-07 0.01965  8.42260E-07 0.05728 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.83210E-03 0.04444  2.37350E-04 0.12823  7.11336E-04 0.09071  4.36744E-04 0.10444  1.09934E-03 0.06612  2.98018E-04 0.11432  4.93102E-05 0.28229 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.15678E-01 0.10438  1.24734E-02 0.00020  3.23122E-02 0.00056  1.06205E-01 0.00348  2.98685E-01 0.00190  1.23608E+00 0.00182  6.12856E+00 0.10832 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.87021E-03 0.04361  2.39692E-04 0.12795  7.15770E-04 0.08722  4.50352E-04 0.10206  1.11466E-03 0.06588  3.01914E-04 0.11393  4.78212E-05 0.29333 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.13089E-01 0.10101  1.24734E-02 0.00020  3.23138E-02 0.00056  1.06171E-01 0.00345  2.98552E-01 0.00186  1.23593E+00 0.00179  6.16697E+00 0.10850 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.65871E+03 0.04060 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.73167E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11799E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32180E-03 0.00673 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.80499E+03 0.00677 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.79440E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.66939E-05 0.00920  6.66758E-05 0.00921  1.20267E-06 0.45113 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.55740E-05 0.02815  5.55252E-05 0.02815  1.38669E-06 0.51713 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.50516E-04 0.02499  1.50555E-04 0.02484  1.38576E-04 0.44438 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73059E+01 0.01164 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50673E+01 0.00026  4.04876E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 21:11:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01883E+00  9.99603E-01  9.91204E-01  9.99657E-01  1.00209E+00  9.96104E-01  9.99843E-01  1.00534E+00  1.00320E+00  9.99284E-01  9.96124E-01  9.93085E-01  1.00006E+00  1.00129E+00  1.00030E+00  1.00318E+00  9.96579E-01  9.94227E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.27271E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67273E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39747E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53458E-01 0.00014  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61367E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51582E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50670E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24367E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66378E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334296 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33396E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33396E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.44399E+03 ;
RUNNING_TIME              (idx, 1)        =  3.73668E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.18090E+00  1.33367E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.71165E+02  5.17380E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.29833E-01  1.90667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.04849E+01  5.40217E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.68307E+02  7.26680E+02 ];
CPU_USAGE                 (idx, 1)        = 9.21670 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98508E+00 5.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.14478E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.72902E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69114E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.97788E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01663E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39704E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67926E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51837E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75425E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.24073E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.12527E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.87527E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.25471E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.35000E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.33193E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.32195E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31094E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51325E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.59601E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.28209E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.17965E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.67565E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84642E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98319E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.73110E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.15850E-02 -2.55793E+27  2.23355E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01881E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.62627E+18 0.00405  1.73537E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  8.81069E+19 0.00052  9.40193E-01 0.00014 ];
U235_FISS                 (idx, [1:   4]) = [  2.26606E+18 0.00338  2.41817E-02 0.00337 ];
PU239_FISS                (idx, [1:   4]) = [  7.73573E+14 0.17350  8.26255E-06 0.17351 ];
PU240_FISS                (idx, [1:   4]) = [  5.17914E+13 0.70600  5.52130E-07 0.70597 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00959E+20 0.00055  7.17290E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28873E+19 0.00139  9.15618E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  8.08713E+17 0.00554  5.74568E-03 0.00552 ];
U238_CAPT                 (idx, [1:   4]) = [  5.16288E+13 0.70601  3.66718E-07 0.70608 ];
PU239_CAPT                (idx, [1:   4]) = [  4.88239E+14 0.22242  3.47070E-06 0.22242 ];
PU240_CAPT                (idx, [1:   4]) = [  2.56974E+13 1.00000  1.84233E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71945E+17 0.01255  1.22174E-03 0.01256 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001879 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.13093E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001879 1.00113E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5456330 5.46172E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3633152 3.63644E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 912397 9.13148E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001879 1.00113E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 4.91738E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36282E+20 3.9E-06  2.36282E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37411E+19 1.1E-07  9.37411E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40797E+20 0.00029  1.26310E+20 0.00020  1.44870E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34538E+20 0.00017  2.20051E+20 0.00011  1.44870E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57703E+20 0.00035  2.57703E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.42989E+22 0.00023  3.25509E+22 0.00021  1.74799E+21 0.00178 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35326E+19 0.00129 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58071E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16263E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47803E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47803E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.79844E-01 0.06304 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.31133E-01 0.01467 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.43947E-04 0.02895 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.65105E+03 0.02575 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08688E-01 0.00012 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.15440E-01 0.04624 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.59288E-01 0.04624 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52058E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99747E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.16565E-01 0.00045  1.01499E-01 0.00044  3.41896E-04 0.01006 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.16625E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.16911E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.16625E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00871E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.98473E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.98394E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.38928E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.39014E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.69403E-01 0.00161 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.70149E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.68093E-03 0.00580  2.82382E-04 0.01838  8.86089E-04 0.01162  6.74496E-04 0.01318  1.44413E-03 0.00872  3.33076E-04 0.01722  6.07487E-05 0.04194 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.63827E-01 0.01438  1.24755E-02 3.2E-05  3.22946E-02 0.00011  1.06394E-01 0.00060  2.98742E-01 0.00027  1.24536E+00 0.00052  5.64510E+00 0.03321 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.32766E-03 0.00790  2.52249E-04 0.02666  8.07509E-04 0.01685  6.15525E-04 0.01798  1.29779E-03 0.01282  3.00061E-04 0.02636  5.45245E-05 0.05970 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.63688E-01 0.02003  1.24758E-02 4.1E-05  3.22945E-02 0.00017  1.06381E-01 0.00085  2.98756E-01 0.00042  1.24518E+00 0.00069  6.54377E+00 0.02579 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.83273E-07 0.00119  8.82800E-07 0.00119  1.02116E-06 0.01676 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.09530E-07 0.00111  8.09098E-07 0.00111  9.35727E-07 0.01674 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.34834E-03 0.01027  2.61736E-04 0.03343  8.09290E-04 0.01944  5.94243E-04 0.02248  1.31562E-03 0.01563  3.10126E-04 0.03003  5.73253E-05 0.07204 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.65984E-01 0.02304  1.24757E-02 5.6E-05  3.23050E-02 0.00020  1.06325E-01 0.00106  2.98765E-01 0.00051  1.24694E+00 0.00114  6.46888E+00 0.03795 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.85109E-07 0.01975  7.84927E-07 0.01975  8.33643E-07 0.05166 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.19908E-07 0.01977  7.19741E-07 0.01977  7.64381E-07 0.05173 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.92089E-03 0.04368  2.55469E-04 0.13324  6.53401E-04 0.08303  5.20033E-04 0.09909  1.14665E-03 0.06877  2.83683E-04 0.11508  6.16590E-05 0.29213 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.93318E-01 0.10246  1.24783E-02 5.8E-05  3.22736E-02 0.00063  1.06013E-01 0.00331  2.98683E-01 0.00192  1.24278E+00 0.00234  7.13694E+00 0.10943 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.90887E-03 0.04339  2.59152E-04 0.12941  6.58500E-04 0.08186  5.16474E-04 0.09953  1.14211E-03 0.06761  2.73794E-04 0.11483  5.88355E-05 0.28553 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.89360E-01 0.10368  1.24783E-02 5.6E-05  3.22716E-02 0.00063  1.06014E-01 0.00329  2.98734E-01 0.00192  1.24315E+00 0.00236  7.13694E+00 0.10943 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.75548E+03 0.03981 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.78014E-07 0.00074 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.04706E-07 0.00057 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38763E-03 0.00574 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.85948E+03 0.00587 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.79698E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68805E-05 0.01017  6.68006E-05 0.01020  1.17586E-06 0.50652 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.83112E-05 0.03051  5.83198E-05 0.03048  7.04935E-07 0.62714 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.44504E-04 0.02739  1.44639E-04 0.02753  1.06426E-04 0.49949 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65896E+01 0.01152 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50670E+01 0.00026  4.02485E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 22:03:46 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01737E+00  1.00142E+00  9.98527E-01  9.98478E-01  9.95935E-01  9.97331E-01  9.90153E-01  1.00802E+00  1.00504E+00  9.98865E-01  9.89976E-01  9.96238E-01  1.00576E+00  9.98262E-01  9.97845E-01  1.00015E+00  9.94332E-01  1.00630E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.28568E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67143E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39809E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53559E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60975E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49750E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48842E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20660E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66566E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334130 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33389E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33389E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.91869E+03 ;
RUNNING_TIME              (idx, 1)        =  4.26151E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.32120E+00  1.40300E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.23463E+02  5.22980E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.49217E-01  1.93833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.59044E+01  5.41947E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.20772E+02  7.32158E+02 ];
CPU_USAGE                 (idx, 1)        = 9.19555 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98554E+00 4.9E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.12120E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.87862E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73175E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.50303E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01967E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39323E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82513E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55875E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.10191E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77930E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.01964E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20650E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04701E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85170E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38534E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38448E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35308E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57409E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.87923E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30904E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26696E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91665E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.42321E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10855E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77619E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.20292E-02 -4.86398E+27  2.25661E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02739E+00 0.00066 ];
TH232_FISS                (idx, [1:   4]) = [  1.60465E+18 0.00400  1.71133E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  8.47587E+19 0.00052  9.03959E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.66847E+18 0.00247  4.97880E-02 0.00237 ];
PU239_FISS                (idx, [1:   4]) = [  1.96121E+16 0.03694  2.09185E-04 0.03691 ];
PU240_FISS                (idx, [1:   4]) = [  5.17922E+14 0.21640  5.53748E-06 0.21640 ];
PU241_FISS                (idx, [1:   4]) = [  4.92806E+14 0.23441  5.26216E-06 0.23440 ];
TH232_CAPT                (idx, [1:   4]) = [  9.87875E+19 0.00051  6.94784E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23906E+19 0.00140  8.71458E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  1.63591E+18 0.00393  1.15057E-02 0.00392 ];
U238_CAPT                 (idx, [1:   4]) = [  2.60170E+14 0.31145  1.83257E-06 0.31146 ];
PU239_CAPT                (idx, [1:   4]) = [  9.54452E+15 0.05148  6.71190E-05 0.05150 ];
PU240_CAPT                (idx, [1:   4]) = [  1.58069E+15 0.13804  1.11209E-05 0.13813 ];
PU241_CAPT                (idx, [1:   4]) = [  5.14173E+13 0.70594  3.56918E-07 0.70594 ];
SM149_CAPT                (idx, [1:   4]) = [  1.69628E+17 0.01234  1.19274E-03 0.01228 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001674 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11810E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001674 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5479945 5.48542E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3614062 3.61737E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907667 9.08399E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001674 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.21655E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36157E+20 4.1E-06  2.36157E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37032E+19 1.0E-07  9.37032E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42148E+20 0.00028  1.27583E+20 0.00020  1.45650E+19 0.00175 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35852E+20 0.00017  2.21287E+20 0.00012  1.45650E+19 0.00175 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59206E+20 0.00035  2.59206E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44066E+22 0.00024  3.26493E+22 0.00022  1.75727E+21 0.00177 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35470E+19 0.00143 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59399E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16467E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42409E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42409E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.16596E-01 0.05988 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.26348E-01 0.01397 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.58031E-04 0.02445 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.94452E+03 0.02880 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09162E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.08274E-01 0.04657 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.53006E-01 0.04657 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52027E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99828E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11623E-01 0.00043  1.00948E-01 0.00042  3.48745E-04 0.00902 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11425E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11110E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11425E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00253E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92397E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92481E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47637E-02 0.00149 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47481E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.84203E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.83848E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85421E-03 0.00544  2.97699E-04 0.02009  9.11773E-04 0.01087  6.93576E-04 0.01156  1.54255E-03 0.00885  3.37061E-04 0.01747  7.15523E-05 0.04068 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.86641E-01 0.01554  1.24761E-02 2.8E-05  3.22718E-02 0.00011  1.06498E-01 0.00054  3.00262E-01 0.00032  1.25552E+00 0.00071  6.10501E+00 0.03025 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.41674E-03 0.00755  2.68493E-04 0.02972  7.94192E-04 0.01585  6.09280E-04 0.01829  1.37866E-03 0.01268  3.05305E-04 0.02627  6.08131E-05 0.05622 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.85062E-01 0.02126  1.24759E-02 3.7E-05  3.22672E-02 0.00016  1.06648E-01 0.00083  3.00264E-01 0.00043  1.25582E+00 0.00089  6.89555E+00 0.02344 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.63654E-07 0.00124  8.63267E-07 0.00124  9.73355E-07 0.01611 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.87289E-07 0.00118  7.86935E-07 0.00118  8.87338E-07 0.01610 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.43517E-03 0.00923  2.66695E-04 0.03236  8.15669E-04 0.01919  6.15847E-04 0.02083  1.37232E-03 0.01424  2.97602E-04 0.03112  6.70347E-05 0.06570 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.93752E-01 0.02585  1.24763E-02 4.5E-05  3.22739E-02 0.00021  1.06433E-01 0.00089  2.99945E-01 0.00054  1.25599E+00 0.00126  6.93716E+00 0.03263 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.60868E-07 0.01963  7.60506E-07 0.01964  8.35792E-07 0.05237 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.93955E-07 0.01963  6.93624E-07 0.01963  7.62319E-07 0.05233 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16944E-03 0.04348  2.31802E-04 0.13342  6.83986E-04 0.08037  5.40243E-04 0.08711  1.38488E-03 0.06477  2.75989E-04 0.12593  5.25373E-05 0.29321 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.73067E-01 0.10527  1.24765E-02 0.00015  3.22649E-02 0.00071  1.07609E-01 0.00402  3.00154E-01 0.00191  1.26112E+00 0.00378  7.06877E+00 0.11665 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.16069E-03 0.04247  2.41322E-04 0.13074  7.01997E-04 0.07799  5.36577E-04 0.08765  1.35586E-03 0.06345  2.74217E-04 0.12149  5.07174E-05 0.28522 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.68669E-01 0.10633  1.24765E-02 0.00015  3.22685E-02 0.00072  1.07584E-01 0.00399  3.00090E-01 0.00190  1.26121E+00 0.00377  7.06877E+00 0.11665 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.18433E+03 0.03902 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.56696E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.80943E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53454E-03 0.00549 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.12528E+03 0.00537 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77125E-09 0.00070 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.67264E-05 0.00812  6.67726E-05 0.00812  1.86280E-06 0.33751 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88889E-05 0.02701  5.88236E-05 0.02716  2.03590E-06 0.41254 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.56887E-04 0.02317  1.56594E-04 0.02321  2.39943E-04 0.33077 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65640E+01 0.01173 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48842E+01 0.00027  3.97059E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 22:56:12 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01387E+00  1.00472E+00  9.88471E-01  1.00210E+00  9.96975E-01  9.99595E-01  1.00456E+00  1.00018E+00  1.00622E+00  1.00264E+00  9.95231E-01  9.95461E-01  9.95196E-01  9.97283E-01  1.00575E+00  9.95814E-01  9.96142E-01  9.99796E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.28721E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67128E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39888E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53640E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60783E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49254E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48350E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19470E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66445E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334483 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33412E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33412E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.39283E+03 ;
RUNNING_TIME              (idx, 1)        =  4.78589E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.46037E+00  1.39167E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.75714E+02  5.22509E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.68683E-01  1.94667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.13348E+01  5.43038E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.73199E+02  7.35482E+02 ];
CPU_USAGE                 (idx, 1)        = 9.17871 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98561E+00 5.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.10240E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.90133E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73778E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.46876E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01228E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38743E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85505E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56520E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.02371E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98924E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.84807E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.37810E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13544E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04350E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.82149E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38542E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35242E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57392E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92623E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73881E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27531E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96787E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.83694E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11766E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.76876E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.71325E-02 -5.99077E+27  2.26788E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02643E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.57644E+18 0.00407  1.68300E-02 0.00400 ];
U233_FISS                 (idx, [1:   4]) = [  8.36302E+19 0.00052  8.92879E-01 0.00016 ];
U235_FISS                 (idx, [1:   4]) = [  5.34400E+18 0.00201  5.70565E-02 0.00198 ];
U238_FISS                 (idx, [1:   4]) = [  2.58893E+13 1.00000  2.74168E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  4.53897E+16 0.02561  4.84531E-04 0.02556 ];
PU240_FISS                (idx, [1:   4]) = [  1.78421E+15 0.11693  1.90462E-05 0.11702 ];
PU241_FISS                (idx, [1:   4]) = [  1.70538E+15 0.12110  1.81839E-05 0.12092 ];
TH232_CAPT                (idx, [1:   4]) = [  9.75472E+19 0.00052  6.86268E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21708E+19 0.00144  8.56257E-02 0.00142 ];
U235_CAPT                 (idx, [1:   4]) = [  1.88945E+18 0.00372  1.32933E-02 0.00373 ];
U238_CAPT                 (idx, [1:   4]) = [  1.81656E+14 0.37418  1.27970E-06 0.37417 ];
PU239_CAPT                (idx, [1:   4]) = [  2.08403E+16 0.03665  1.46576E-04 0.03662 ];
PU240_CAPT                (idx, [1:   4]) = [  4.32136E+15 0.07767  3.04117E-05 0.07773 ];
PU241_CAPT                (idx, [1:   4]) = [  4.12059E+14 0.24366  2.90074E-06 0.24366 ];
SM149_CAPT                (idx, [1:   4]) = [  1.70318E+17 0.01191  1.19808E-03 0.01187 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002364 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10927E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002364 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5484100 5.48902E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3613891 3.61697E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 904373 9.05098E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002364 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.58325E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36146E+20 4.5E-06  2.36146E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36891E+19 1.1E-07  9.36891E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42146E+20 0.00029  1.27568E+20 0.00019  1.45777E+19 0.00199 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35835E+20 0.00017  2.21258E+20 0.00011  1.45777E+19 0.00199 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.58959E+20 0.00037  2.58959E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43533E+22 0.00026  3.25917E+22 0.00023  1.76165E+21 0.00193 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.34393E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59275E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16228E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39598E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39598E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.14682E-01 0.06062 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.41039E-01 0.01341 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.50008E-04 0.02794 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.92824E+03 0.02330 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09493E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.27856E-01 0.04464 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.71069E-01 0.04464 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52052E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99858E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11564E-01 0.00044  1.00956E-01 0.00043  3.42517E-04 0.00895 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11827E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11943E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11827E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00256E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90245E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90359E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50841E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50648E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.89260E-01 0.00159 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.88690E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.83071E-03 0.00553  2.91198E-04 0.02082  8.99298E-04 0.01120  6.88293E-04 0.01283  1.53512E-03 0.00847  3.41646E-04 0.01813  7.51567E-05 0.03650 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.99679E-01 0.01466  1.24767E-02 3.4E-05  3.22602E-02 0.00013  1.06447E-01 0.00053  3.00517E-01 0.00031  1.25680E+00 0.00070  6.37124E+00 0.02669 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43438E-03 0.00768  2.55629E-04 0.02967  8.08568E-04 0.01579  6.10141E-04 0.01786  1.38859E-03 0.01154  3.06463E-04 0.02796  6.49852E-05 0.05240 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.92526E-01 0.01939  1.24776E-02 7.3E-05  3.22522E-02 0.00020  1.06424E-01 0.00075  3.00498E-01 0.00045  1.25741E+00 0.00093  6.98497E+00 0.02204 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.54741E-07 0.00125  8.54307E-07 0.00126  9.78583E-07 0.01863 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.79103E-07 0.00117  7.78709E-07 0.00117  8.91883E-07 0.01857 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39279E-03 0.00906  2.55142E-04 0.03387  7.91447E-04 0.01975  6.10960E-04 0.02103  1.37380E-03 0.01402  2.94791E-04 0.03292  6.66448E-05 0.06324 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.97995E-01 0.02458  1.24774E-02 8.6E-05  3.22549E-02 0.00023  1.06407E-01 0.00088  3.00555E-01 0.00049  1.25696E+00 0.00142  7.05875E+00 0.03101 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.54074E-07 0.01963  7.53849E-07 0.01963  8.45092E-07 0.05361 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.87784E-07 0.01964  6.87578E-07 0.01964  7.71202E-07 0.05370 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.99522E-03 0.04333  2.20725E-04 0.13940  7.39414E-04 0.08268  5.66149E-04 0.07861  1.16413E-03 0.06919  2.64201E-04 0.12035  4.06038E-05 0.26932 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.60831E-01 0.08376  1.24762E-02 0.00017  3.22716E-02 0.00082  1.06304E-01 0.00274  3.00936E-01 0.00201  1.25811E+00 0.00343  8.00828E+00 0.07236 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97531E-03 0.04291  2.15903E-04 0.13692  7.27818E-04 0.08071  5.65589E-04 0.07664  1.15902E-03 0.06839  2.67192E-04 0.12009  3.97882E-05 0.26305 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.60987E-01 0.08516  1.24762E-02 0.00017  3.22723E-02 0.00082  1.06310E-01 0.00273  3.00981E-01 0.00201  1.25788E+00 0.00341  8.00828E+00 0.07236 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.00065E+03 0.03938 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.48466E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.73389E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.42246E-03 0.00510 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03502E+03 0.00526 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.74474E-09 0.00080 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68508E-05 0.01008  6.68615E-05 0.01004  1.23216E-06 0.42677 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.47467E-05 0.02872  5.48004E-05 0.02894  1.33075E-06 0.58171 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.48805E-04 0.02738  1.48775E-04 0.02748  1.57947E-04 0.40593 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65502E+01 0.01168 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48350E+01 0.00027  3.95158E+01 0.00036 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Tue Apr 26 23:50:06 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01255E+00  1.00771E+00  9.95769E-01  9.98468E-01  1.00300E+00  1.00004E+00  9.98400E-01  9.97954E-01  9.95411E-01  1.00092E+00  9.95244E-01  9.99492E-01  9.98370E-01  1.00201E+00  9.92520E-01  9.99007E-01  1.00341E+00  9.99718E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.29198E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.67080E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39733E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53502E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61042E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48185E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47277E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.18004E+01 0.00033  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.66528E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334387 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33393E+04 0.00052 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33393E+04 0.00052 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.87766E+03 ;
RUNNING_TIME              (idx, 1)        =  5.32482E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.60110E+00  1.40733E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.29416E+02  5.37020E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.88450E-01  1.97667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.71895E+01  5.85458E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.26669E+02  7.36314E+02 ];
CPU_USAGE                 (idx, 1)        = 9.16024 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97540E+00 0.00015 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.08180E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.90124E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73842E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.08254E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99746E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37701E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86976E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56687E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.08379E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.16964E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.82823E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.54318E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21228E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20679E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20608E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37708E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34487E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56425E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95018E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.12064E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26754E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99190E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.15794E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09692E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77131E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.22914E-02 -7.12984E+27  2.27927E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02528E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.56028E+18 0.00422  1.66667E-02 0.00411 ];
U233_FISS                 (idx, [1:   4]) = [  8.27621E+19 0.00053  8.84132E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  5.84631E+18 0.00223  6.24541E-02 0.00215 ];
PU239_FISS                (idx, [1:   4]) = [  8.25546E+16 0.01704  8.81911E-04 0.01704 ];
PU240_FISS                (idx, [1:   4]) = [  3.49730E+15 0.08666  3.73775E-05 0.08671 ];
PU241_FISS                (idx, [1:   4]) = [  3.88554E+15 0.09126  4.15113E-05 0.09130 ];
TH232_CAPT                (idx, [1:   4]) = [  9.66589E+19 0.00052  6.79795E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20579E+19 0.00153  8.48033E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.07210E+18 0.00345  1.45727E-02 0.00342 ];
U238_CAPT                 (idx, [1:   4]) = [  2.32687E+14 0.32885  1.63873E-06 0.32886 ];
PU239_CAPT                (idx, [1:   4]) = [  3.87474E+16 0.02563  2.72525E-04 0.02561 ];
PU240_CAPT                (idx, [1:   4]) = [  9.94212E+15 0.05293  6.99422E-05 0.05294 ];
PU241_CAPT                (idx, [1:   4]) = [  8.28267E+14 0.17845  5.83021E-06 0.17854 ];
SM149_CAPT                (idx, [1:   4]) = [  1.75902E+17 0.01238  1.23716E-03 0.01239 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001800 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09415E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001800 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5483795 5.48902E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3610527 3.61363E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907478 9.08285E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001800 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.32248E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36153E+20 4.4E-06  2.36153E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36765E+19 9.4E-08  9.36765E+19 9.4E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42135E+20 0.00028  1.27590E+20 0.00020  1.45452E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35812E+20 0.00017  2.21266E+20 0.00011  1.45452E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59044E+20 0.00037  2.59044E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43132E+22 0.00025  3.25559E+22 0.00022  1.75727E+21 0.00174 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35294E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59341E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15985E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.36855E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.36855E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.92804E-01 0.05974 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.53057E-01 0.01348 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.49634E-04 0.02638 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.21108E+03 0.02745 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09174E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.04546E-01 0.04689 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.49625E-01 0.04689 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52095E+00 4.3E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99885E+02 9.4E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.10942E-01 0.00039  1.00876E-01 0.00039  3.44826E-04 0.00830 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11604E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11672E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11604E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00267E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88519E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88679E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53469E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53200E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.92594E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.92400E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.84735E-03 0.00519  2.90659E-04 0.01957  8.93065E-04 0.01023  6.97505E-04 0.01245  1.54571E-03 0.00858  3.39561E-04 0.01694  8.08444E-05 0.03975 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.04052E-01 0.01521  1.24764E-02 3.9E-05  3.22484E-02 0.00013  1.06705E-01 0.00058  3.00587E-01 0.00029  1.25833E+00 0.00080  5.98580E+00 0.02950 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43780E-03 0.00746  2.59895E-04 0.02794  8.03994E-04 0.01548  6.23393E-04 0.01880  1.37994E-03 0.01161  3.04714E-04 0.02537  6.58638E-05 0.05280 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.94037E-01 0.01960  1.24767E-02 5.2E-05  3.22414E-02 0.00019  1.06764E-01 0.00079  3.00670E-01 0.00044  1.25601E+00 0.00117  6.99488E+00 0.02053 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.50115E-07 0.00126  8.49727E-07 0.00126  9.65836E-07 0.01676 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.74366E-07 0.00118  7.74013E-07 0.00118  8.79889E-07 0.01678 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40193E-03 0.00825  2.68364E-04 0.03208  7.86481E-04 0.01874  6.12867E-04 0.02083  1.36406E-03 0.01431  2.97224E-04 0.03049  7.29313E-05 0.06730 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.04931E-01 0.02659  1.24769E-02 4.5E-05  3.22465E-02 0.00024  1.06594E-01 0.00093  3.00430E-01 0.00051  1.25576E+00 0.00150  6.83764E+00 0.03195 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.49238E-07 0.01959  7.48867E-07 0.01960  7.81411E-07 0.05425 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.82662E-07 0.01960  6.82323E-07 0.01960  7.12081E-07 0.05420 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16198E-03 0.04241  2.40602E-04 0.13297  6.21463E-04 0.07753  5.91505E-04 0.08642  1.35894E-03 0.06181  2.89362E-04 0.12743  6.01137E-05 0.29362 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.03990E-01 0.10264  1.24761E-02 0.00015  3.22254E-02 0.00071  1.07167E-01 0.00353  2.99752E-01 0.00185  1.25285E+00 0.00364  8.02912E+00 0.09062 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.17605E-03 0.04149  2.46355E-04 0.13157  6.30848E-04 0.07724  5.87256E-04 0.08437  1.35737E-03 0.05986  2.94318E-04 0.12595  5.99063E-05 0.28822 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.07575E-01 0.10580  1.24762E-02 0.00015  3.22263E-02 0.00071  1.07120E-01 0.00349  2.99765E-01 0.00185  1.25254E+00 0.00362  8.01565E+00 0.09121 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.24530E+03 0.03840 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.44301E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.69071E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.43668E-03 0.00604 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.07103E+03 0.00609 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.74559E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.59390E-05 0.01001  6.59547E-05 0.01002  8.65091E-07 0.50352 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.58046E-05 0.03131  5.58004E-05 0.03131  7.43550E-07 0.53441 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.48925E-04 0.02550  1.49097E-04 0.02549  1.05984E-04 0.50245 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61741E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47277E+01 0.00027  3.93439E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 00:45:07 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01805E+00  1.00218E+00  1.00177E+00  9.98601E-01  9.95867E-01  9.94857E-01  9.99267E-01  9.97224E-01  1.00891E+00  9.99992E-01  9.93108E-01  9.95974E-01  1.00300E+00  9.96234E-01  9.98850E-01  9.97503E-01  9.99581E-01  9.99037E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.31263E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66874E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39203E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53046E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61562E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46759E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45848E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17017E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.67741E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334061 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33410E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33410E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.37094E+03 ;
RUNNING_TIME              (idx, 1)        =  5.87507E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.74380E+00  1.42700E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.84245E+02  5.48294E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.06783E-01  1.83333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.33589E+01  6.16930E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.81378E+02  7.42967E+02 ];
CPU_USAGE                 (idx, 1)        = 9.14193 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.96997E+00 0.00017 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.06137E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.92244E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74436E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.35832E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96283E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35628E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92560E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57491E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.59886E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.74818E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.10781E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03531E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44256E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70331E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34462E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37116E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33746E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55581E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.98889E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27295E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26335E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04832E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.40403E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07324E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.79363E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.28311E-02 -1.16650E+28  2.32462E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02213E+00 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.52243E+18 0.00418  1.62706E-02 0.00412 ];
U233_FISS                 (idx, [1:   4]) = [  8.07101E+19 0.00054  8.62602E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  6.86387E+18 0.00190  7.33578E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  2.55710E+13 1.00000  2.77408E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  2.88017E+17 0.00944  3.07830E-03 0.00941 ];
PU240_FISS                (idx, [1:   4]) = [  2.09855E+16 0.03424  2.24229E-04 0.03421 ];
PU241_FISS                (idx, [1:   4]) = [  3.08092E+16 0.02758  3.29209E-04 0.02752 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44907E+19 0.00052  6.61631E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17863E+19 0.00154  8.25272E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  2.42043E+18 0.00356  1.69486E-02 0.00357 ];
U238_CAPT                 (idx, [1:   4]) = [  1.19766E+15 0.15556  8.38228E-06 0.15556 ];
PU239_CAPT                (idx, [1:   4]) = [  1.38563E+17 0.01364  9.70156E-04 0.01360 ];
PU240_CAPT                (idx, [1:   4]) = [  5.54807E+16 0.02229  3.88470E-04 0.02226 ];
PU241_CAPT                (idx, [1:   4]) = [  6.28592E+15 0.06786  4.40083E-05 0.06788 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74147E+17 0.01206  1.21938E-03 0.01205 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002298 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08399E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002298 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5492548 5.49741E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3598658 3.60166E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911092 9.11772E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002298 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.63913E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36300E+20 4.6E-06  2.36300E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36374E+19 9.4E-08  9.36374E+19 9.4E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42814E+20 0.00029  1.28149E+20 0.00018  1.46654E+19 0.00207 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36451E+20 0.00018  2.21786E+20 0.00010  1.46654E+19 0.00207 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59788E+20 0.00035  2.59788E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44080E+22 0.00025  3.26440E+22 0.00022  1.76405E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36874E+19 0.00142 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60139E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15947E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26454E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26454E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.44366E-01 0.05244 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21221E-01 0.01346 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.72417E-04 0.02606 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.96966E+03 0.02693 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08825E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.83411E-01 0.03937 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.21061E-01 0.03937 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52356E+00 4.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99968E+02 9.4E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.08784E-01 0.00043  1.00640E-01 0.00043  3.48571E-04 0.00885 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09359E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09622E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09359E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00056E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84471E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84349E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.59807E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.59976E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.01584E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.02158E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.90285E-03 0.00552  2.84703E-04 0.01908  9.21401E-04 0.01076  7.08163E-04 0.01187  1.54451E-03 0.00846  3.63733E-04 0.01724  8.03419E-05 0.03663 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.09097E-01 0.01484  1.24768E-02 5.1E-05  3.22328E-02 0.00014  1.06734E-01 0.00051  3.01181E-01 0.00031  1.26168E+00 0.00092  6.28264E+00 0.02617 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47828E-03 0.00710  2.48258E-04 0.02817  8.22078E-04 0.01452  6.21571E-04 0.01716  1.39445E-03 0.01188  3.22766E-04 0.02591  6.91654E-05 0.05484 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.04286E-01 0.02196  1.24776E-02 7.6E-05  3.22238E-02 0.00020  1.06772E-01 0.00074  3.01146E-01 0.00043  1.26232E+00 0.00118  6.83589E+00 0.02146 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.34909E-07 0.00120  8.34586E-07 0.00119  9.30524E-07 0.01649 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.58713E-07 0.00113  7.58419E-07 0.00112  8.45634E-07 0.01651 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.44998E-03 0.00911  2.50790E-04 0.03320  8.07875E-04 0.01864  6.21861E-04 0.02135  1.36532E-03 0.01431  3.35017E-04 0.02829  6.91179E-05 0.06130 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08239E-01 0.02345  1.24745E-02 6.9E-05  3.22200E-02 0.00025  1.06770E-01 0.00096  3.01314E-01 0.00054  1.26129E+00 0.00152  6.81020E+00 0.03079 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.43664E-07 0.01967  7.43466E-07 0.01966  7.92118E-07 0.08334 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.75957E-07 0.01966  6.75779E-07 0.01966  7.19215E-07 0.08286 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.91604E-03 0.04462  2.27134E-04 0.14021  6.30206E-04 0.08194  5.55502E-04 0.09619  1.13659E-03 0.06277  2.97735E-04 0.13074  6.88773E-05 0.25468 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.94446E-01 0.08091  1.24749E-02 0.00017  3.22461E-02 0.00082  1.07120E-01 0.00362  3.01368E-01 0.00199  1.26855E+00 0.00415  6.38457E+00 0.09549 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89521E-03 0.04390  2.29948E-04 0.14139  6.11824E-04 0.07835  5.54464E-04 0.09332  1.12947E-03 0.06233  2.99879E-04 0.13089  6.96263E-05 0.24874 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.04122E-01 0.08399  1.24749E-02 0.00017  3.22476E-02 0.00082  1.07107E-01 0.00360  3.01342E-01 0.00198  1.26840E+00 0.00415  6.39092E+00 0.09529 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.94805E+03 0.04071 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.29042E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.53373E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40284E-03 0.00665 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.10533E+03 0.00673 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77048E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77365E-05 0.00916  6.77229E-05 0.00918  1.16761E-06 0.45823 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.71987E-05 0.02672  5.74780E-05 0.02763  9.43552E-07 0.52667 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.71373E-04 0.02423  1.71541E-04 0.02433  1.31941E-04 0.44558 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60868E+01 0.01035 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45848E+01 0.00026  3.89519E+01 0.00035 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 01:40:18 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01069E+00  1.00156E+00  9.96743E-01  1.00133E+00  9.99447E-01  9.96909E-01  1.00315E+00  1.00302E+00  1.00468E+00  1.00072E+00  9.90755E-01  9.99276E-01  9.95870E-01  9.94508E-01  1.00231E+00  1.00238E+00  9.93131E-01  1.00351E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.36055E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.66395E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37815E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51832E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62792E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46232E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45316E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20393E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.71638E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333665 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33382E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33382E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.86656E+03 ;
RUNNING_TIME              (idx, 1)        =  6.42684E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.89000E+00  1.46200E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.39223E+02  5.49781E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.24750E-01  1.79667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.94349E+01  6.07595E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.36649E+02  7.46637E+02 ];
CPU_USAGE                 (idx, 1)        = 9.12821 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97108E+00 0.00017 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.04611E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.91198E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73934E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.14999E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92256E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33633E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95543E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57191E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53436E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38932E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.35969E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14081E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69246E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23782E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58134E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34516E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31651E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52778E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02218E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59805E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23302E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02919E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.13976E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01513E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.82142E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06334E-01 -2.34782E+28  2.44275E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01851E+00 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.51345E+18 0.00431  1.61655E-02 0.00420 ];
U233_FISS                 (idx, [1:   4]) = [  7.93910E+19 0.00058  8.48073E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.13111E+18 0.00201  7.61756E-02 0.00193 ];
U238_FISS                 (idx, [1:   4]) = [  2.60065E+14 0.31145  2.76927E-06 0.31146 ];
PU239_FISS                (idx, [1:   4]) = [  6.02719E+17 0.00640  6.43820E-03 0.00636 ];
PU240_FISS                (idx, [1:   4]) = [  6.90538E+16 0.01942  7.37543E-04 0.01940 ];
PU241_FISS                (idx, [1:   4]) = [  1.22835E+17 0.01388  1.31233E-03 0.01389 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27663E+19 0.00050  6.46622E-01 0.00032 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15870E+19 0.00146  8.07677E-02 0.00145 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48901E+18 0.00309  1.73499E-02 0.00309 ];
U238_CAPT                 (idx, [1:   4]) = [  2.60641E+15 0.09430  1.81609E-05 0.09425 ];
PU239_CAPT                (idx, [1:   4]) = [  2.76645E+17 0.00969  1.92824E-03 0.00968 ];
PU240_CAPT                (idx, [1:   4]) = [  1.86458E+17 0.01238  1.29957E-03 0.01234 ];
PU241_CAPT                (idx, [1:   4]) = [  2.53374E+16 0.03102  1.76682E-04 0.03104 ];
SM149_CAPT                (idx, [1:   4]) = [  1.73792E+17 0.01197  1.21138E-03 0.01196 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001472 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07940E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001472 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5497461 5.50276E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3587485 3.59067E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 916526 9.17365E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001472 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.23517E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36605E+20 4.9E-06  2.36605E+20 4.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35966E+19 9.1E-08  9.35966E+19 9.1E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43444E+20 0.00029  1.28413E+20 0.00019  1.50306E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37041E+20 0.00018  2.22010E+20 0.00011  1.50306E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60714E+20 0.00034  2.60714E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47173E+22 0.00024  3.29253E+22 0.00021  1.79198E+21 0.00189 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39177E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60958E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16220E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.05311E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.05311E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.57562E-01 0.05313 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.17538E-01 0.01173 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.15792E-04 0.02107 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.79199E+03 0.02776 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08266E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.96625E-01 0.03816 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.32770E-01 0.03816 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52793E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00056E+02 9.1E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07540E-01 0.00046  1.00510E-01 0.00045  3.41945E-04 0.00871 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07670E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.07559E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07670E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99360E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81466E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81461E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64686E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64663E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09128E-01 0.00142 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.09252E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.86839E-03 0.00539  2.85454E-04 0.01923  8.99105E-04 0.01063  6.91405E-04 0.01329  1.55495E-03 0.00784  3.57982E-04 0.01727  7.94925E-05 0.03664 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10783E-01 0.01402  1.24796E-02 9.6E-05  3.21954E-02 0.00015  1.06849E-01 0.00053  3.01272E-01 0.00031  1.25939E+00 0.00102  6.39351E+00 0.02729 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.41024E-03 0.00766  2.43593E-04 0.03029  7.94844E-04 0.01555  6.15365E-04 0.01770  1.36262E-03 0.01248  3.17903E-04 0.02598  7.59096E-05 0.05385 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.18900E-01 0.02044  1.24814E-02 0.00017  3.21831E-02 0.00023  1.06760E-01 0.00073  3.01415E-01 0.00043  1.25984E+00 0.00138  7.07357E+00 0.02136 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.30972E-07 0.00130  8.30474E-07 0.00130  9.74793E-07 0.01940 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.54092E-07 0.00121  7.53640E-07 0.00121  8.84564E-07 0.01937 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.39492E-03 0.00873  2.46286E-04 0.03518  7.80778E-04 0.01838  6.08305E-04 0.02182  1.38556E-03 0.01354  3.02393E-04 0.02931  7.15903E-05 0.06232 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.16535E-01 0.02631  1.24793E-02 0.00011  3.21938E-02 0.00028  1.06742E-01 0.00092  3.01362E-01 0.00053  1.25865E+00 0.00237  7.13354E+00 0.03034 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.37196E-07 0.01964  7.36854E-07 0.01964  8.40193E-07 0.05648 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.69238E-07 0.01964  6.68929E-07 0.01964  7.62555E-07 0.05647 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.94844E-03 0.04413  2.24022E-04 0.16185  6.58638E-04 0.08413  5.32096E-04 0.09007  1.13760E-03 0.06539  3.20486E-04 0.12581  7.55996E-05 0.25134 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.50551E-01 0.11605  1.24742E-02 0.00022  3.21510E-02 0.00095  1.07259E-01 0.00384  3.01968E-01 0.00205  1.26250E+00 0.00527  7.39139E+00 0.08607 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.93916E-03 0.04376  2.19669E-04 0.16051  6.65103E-04 0.08202  5.26139E-04 0.08926  1.13968E-03 0.06446  3.16492E-04 0.12699  7.20802E-05 0.24765 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.46818E-01 0.11500  1.24742E-02 0.00022  3.21439E-02 0.00098  1.07252E-01 0.00384  3.01882E-01 0.00203  1.26181E+00 0.00524  7.42806E+00 0.08469 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.02058E+03 0.03980 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.24536E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.48251E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.35356E-03 0.00584 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.06784E+03 0.00587 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.87164E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84136E-05 0.00668  6.83808E-05 0.00676  1.48825E-06 0.43741 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.79832E-05 0.02299  5.79329E-05 0.02316  1.40256E-06 0.49372 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.15397E-04 0.01979  2.15631E-04 0.01987  1.53216E-04 0.40547 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62081E+01 0.01080 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45316E+01 0.00029  3.86708E+01 0.00039 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 02:35:36 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01621E+00  1.00363E+00  9.99690E-01  9.99122E-01  9.99303E-01  9.97108E-01  1.00082E+00  1.00234E+00  9.97201E-01  1.00065E+00  9.91435E-01  9.89882E-01  1.00463E+00  9.98382E-01  9.98794E-01  1.00054E+00  9.97618E-01  1.00265E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.40347E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65965E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36356E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50536E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64257E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46914E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45989E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26322E+01 0.00033  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.75678E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333621 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33366E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33366E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.36401E+03 ;
RUNNING_TIME              (idx, 1)        =  6.97985E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.04892E+00  1.58917E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.94305E+02  5.50814E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.43500E-01  1.87500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.60167E-02  1.37167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.54332E+01  5.99823E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.92038E+02  7.47194E+02 ];
CPU_USAGE                 (idx, 1)        = 9.11769 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97521E+00 0.00013 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.03444E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.90901E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73742E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.36159E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91209E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33131E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96281E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57037E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64031E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56432E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.45813E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.15998E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76416E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39300E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93016E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33719E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31060E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51955E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.03924E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00675E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22304E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01748E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.34141E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99794E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.83377E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61218E-01 -3.55966E+28  2.56394E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01610E+00 0.00077 ];
TH232_FISS                (idx, [1:   4]) = [  1.50747E+18 0.00427  1.61136E-02 0.00420 ];
U233_FISS                 (idx, [1:   4]) = [  7.91908E+19 0.00060  8.46532E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.10330E+18 0.00185  7.59343E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  2.89048E+14 0.29644  3.09542E-06 0.29645 ];
PU239_FISS                (idx, [1:   4]) = [  6.56436E+17 0.00616  7.01692E-03 0.00612 ];
PU240_FISS                (idx, [1:   4]) = [  8.45332E+16 0.01744  9.03708E-04 0.01744 ];
PU241_FISS                (idx, [1:   4]) = [  1.48578E+17 0.01308  1.58836E-03 0.01308 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22718E+19 0.00056  6.42247E-01 0.00036 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15321E+19 0.00143  8.02683E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  2.47811E+18 0.00336  1.72484E-02 0.00332 ];
U238_CAPT                 (idx, [1:   4]) = [  2.69057E+15 0.09903  1.87162E-05 0.09897 ];
PU239_CAPT                (idx, [1:   4]) = [  3.07407E+17 0.00916  2.13979E-03 0.00918 ];
PU240_CAPT                (idx, [1:   4]) = [  2.24842E+17 0.01138  1.56523E-03 0.01142 ];
PU241_CAPT                (idx, [1:   4]) = [  3.20609E+16 0.02763  2.23160E-04 0.02764 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81586E+17 0.01163  1.26387E-03 0.01160 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000984 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06589E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000984 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5496519 5.50201E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3579110 3.58247E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 925355 9.26177E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000984 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.77534E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36674E+20 4.8E-06  2.36674E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35900E+19 1.0E-07  9.35900E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43627E+20 0.00029  1.28144E+20 0.00020  1.54828E+19 0.00197 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37217E+20 0.00017  2.21734E+20 0.00011  1.54828E+19 0.00197 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61126E+20 0.00035  2.61126E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50351E+22 0.00024  3.32109E+22 0.00022  1.82425E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41860E+19 0.00158 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61403E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16577E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87044E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87044E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.70795E-01 0.04428 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.01976E-01 0.01016 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.67627E-04 0.02009 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.07349E+03 0.02926 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07387E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99995E-01 7.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.85416E-01 0.03012 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.12672E-01 0.03012 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52883E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05825E-01 0.00045  1.00304E-01 0.00044  3.52128E-04 0.00896 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06383E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.06392E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06383E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98902E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81206E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81282E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65117E-02 0.00148 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64959E-02 0.00090 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09703E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10220E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.96025E-03 0.00529  2.81362E-04 0.01874  9.05984E-04 0.01062  7.11471E-04 0.01181  1.60812E-03 0.00842  3.71714E-04 0.01745  8.15979E-05 0.03613 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13663E-01 0.01425  1.24807E-02 0.00011  3.21922E-02 0.00016  1.06727E-01 0.00057  3.01295E-01 0.00030  1.25830E+00 0.00104  6.47099E+00 0.02522 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47289E-03 0.00742  2.38009E-04 0.02781  8.11287E-04 0.01440  6.19728E-04 0.01794  1.41131E-03 0.01114  3.23882E-04 0.02514  6.86691E-05 0.05062 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.05479E-01 0.01863  1.24795E-02 0.00011  3.21847E-02 0.00020  1.06716E-01 0.00079  3.01155E-01 0.00041  1.25645E+00 0.00146  6.95734E+00 0.02115 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.38562E-07 0.00129  8.38134E-07 0.00128  9.68206E-07 0.02452 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.59549E-07 0.00122  7.59160E-07 0.00122  8.77159E-07 0.02461 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.49349E-03 0.00906  2.46634E-04 0.03280  7.97686E-04 0.01826  6.19166E-04 0.02000  1.43125E-03 0.01404  3.24277E-04 0.02994  7.44696E-05 0.06185 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.14302E-01 0.02476  1.24799E-02 0.00016  3.21898E-02 0.00026  1.06742E-01 0.00095  3.01327E-01 0.00051  1.25718E+00 0.00187  6.88743E+00 0.02981 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.35318E-07 0.01964  7.35023E-07 0.01964  7.90550E-07 0.05057 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.66286E-07 0.01964  6.66018E-07 0.01964  7.16425E-07 0.05060 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.39427E-03 0.04560  2.67420E-04 0.13697  7.79899E-04 0.07945  6.65459E-04 0.09678  1.30114E-03 0.06469  3.32869E-04 0.10850  4.74902E-05 0.24869 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.65421E-01 0.09102  1.24757E-02 0.00015  3.22430E-02 0.00100  1.06493E-01 0.00270  3.01289E-01 0.00192  1.25597E+00 0.00523  6.28634E+00 0.11331 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.38890E-03 0.04476  2.68340E-04 0.13490  7.79349E-04 0.07803  6.72438E-04 0.09430  1.29447E-03 0.06425  3.29853E-04 0.10608  4.44493E-05 0.24710 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.57089E-01 0.08688  1.24758E-02 0.00015  3.22415E-02 0.00101  1.06478E-01 0.00268  3.01411E-01 0.00193  1.25576E+00 0.00511  6.28090E+00 0.11311 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.63462E+03 0.04143 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.30790E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.52511E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.60324E-03 0.00621 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.33718E+03 0.00617 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99496E-09 0.00094 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.82616E-05 0.00599  6.82723E-05 0.00600  2.00711E-06 0.33195 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.76598E-05 0.02151  5.76709E-05 0.02153  1.30004E-06 0.43970 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.64147E-04 0.01966  2.64299E-04 0.01970  2.34544E-04 0.33055 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.58213E+01 0.01065 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45989E+01 0.00029  3.86371E+01 0.00038 ];


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
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'insolente.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 14:57:37 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 03:06:05 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650995857115 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02236E+00  9.99833E-01  1.00130E+00  9.96928E-01  9.98466E-01  9.89417E-01  1.00002E+00  1.00599E+00  1.00746E+00  9.94380E-01  9.99162E-01  9.97491E-01  9.99113E-01  9.97471E-01  9.97305E-01  9.96085E-01  9.98089E-01  9.99127E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.45715E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65428E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.34915E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49288E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.65994E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47629E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46697E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.32190E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.80573E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334026 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33399E+04 0.00059 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33399E+04 0.00059 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.63636E+03 ;
RUNNING_TIME              (idx, 1)        =  7.28475E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.09477E+00  1.09477E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.19828E+00  1.49367E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.24583E+02  3.02778E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.62767E-01  1.92667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  3.97500E-02  1.37333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.88803E+01  3.44705E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.25069E+02  7.25069E+02 ];
CPU_USAGE                 (idx, 1)        = 9.10994 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.97010E+00 0.00013 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.02627E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.93 ;
ALLOC_MEMSIZE             (idx, 1)        = 22071.64;
MEMSIZE                   (idx, 1)        = 21881.11;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 190.53;

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

TOT_ACTIVITY              (idx, 1)        =  6.90645E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73706E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59189E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90574E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32757E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96656E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57034E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65464E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61270E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47019E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16112E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78543E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43997E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03094E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33610E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30983E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51846E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04261E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13387E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21845E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01530E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.37970E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98684E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.84494E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16420E-01 -4.77849E+28  2.68582E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01191E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.49896E+18 0.00419  1.60140E-02 0.00409 ];
U233_FISS                 (idx, [1:   4]) = [  7.92644E+19 0.00055  8.46879E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.09028E+18 0.00200  7.57518E-02 0.00188 ];
U238_FISS                 (idx, [1:   4]) = [  2.62783E+14 0.31145  2.81170E-06 0.31145 ];
PU239_FISS                (idx, [1:   4]) = [  6.56169E+17 0.00634  7.01112E-03 0.00635 ];
PU240_FISS                (idx, [1:   4]) = [  8.89658E+16 0.01642  9.50540E-04 0.01641 ];
PU241_FISS                (idx, [1:   4]) = [  1.51596E+17 0.01291  1.61945E-03 0.01288 ];
TH232_CAPT                (idx, [1:   4]) = [  9.19164E+19 0.00050  6.39158E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15627E+19 0.00144  8.04045E-02 0.00141 ];
U235_CAPT                 (idx, [1:   4]) = [  2.47421E+18 0.00337  1.72041E-02 0.00330 ];
U238_CAPT                 (idx, [1:   4]) = [  2.69095E+15 0.10079  1.87214E-05 0.10082 ];
PU239_CAPT                (idx, [1:   4]) = [  3.01307E+17 0.00906  2.09524E-03 0.00905 ];
PU240_CAPT                (idx, [1:   4]) = [  2.25514E+17 0.01145  1.56807E-03 0.01143 ];
PU241_CAPT                (idx, [1:   4]) = [  3.32366E+16 0.02581  2.31068E-04 0.02580 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84405E+17 0.01163  1.28222E-03 0.01160 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001965 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08799E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001965 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494405 5.49947E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3576092 3.57924E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 931468 9.32167E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001965 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.17929E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 2.4E-09  3.52484E-02 2.4E-09  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36684E+20 5.1E-06  2.36684E+20 5.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35892E+19 1.1E-07  9.35892E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43752E+20 0.00028  1.27866E+20 0.00019  1.58858E+19 0.00177 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37341E+20 0.00017  2.21455E+20 0.00011  1.58858E+19 0.00177 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61498E+20 0.00034  2.61498E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53524E+22 0.00024  3.35017E+22 0.00021  1.85064E+21 0.00175 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43769E+19 0.00157 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61718E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16932E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.70924E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.70924E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.40965E-01 0.04231 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.09117E-01 0.00939 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.16515E-04 0.01866 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.12852E+03 0.03329 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06790E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 9.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.45193E-01 0.02461 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.66365E-01 0.02462 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52896E+00 5.1E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05067E-01 0.00043  1.00231E-01 0.00043  3.42892E-04 0.00905 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05335E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05138E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05335E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98426E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81596E+00 0.00023 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81423E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64480E-02 0.00156 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64727E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09966E-01 0.00154 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10247E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.91883E-03 0.00539  2.73251E-04 0.01930  9.19285E-04 0.01117  6.90263E-04 0.01281  1.58902E-03 0.00793  3.64584E-04 0.01695  8.24265E-05 0.03657 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.10763E-01 0.01463  1.24812E-02 0.00012  3.22026E-02 0.00015  1.06780E-01 0.00058  3.01357E-01 0.00030  1.25666E+00 0.00113  6.34922E+00 0.02575 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.42070E-03 0.00783  2.35489E-04 0.02979  8.10655E-04 0.01534  6.03904E-04 0.01782  1.38450E-03 0.01161  3.20713E-04 0.02590  6.54390E-05 0.05481 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00156E-01 0.02001  1.24813E-02 0.00015  3.22017E-02 0.00019  1.06800E-01 0.00077  3.01343E-01 0.00042  1.25788E+00 0.00149  6.88840E+00 0.02188 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.50093E-07 0.00147  8.49574E-07 0.00146  9.94282E-07 0.03161 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.69350E-07 0.00141  7.68879E-07 0.00140  9.00004E-07 0.03176 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41234E-03 0.00923  2.41640E-04 0.03280  7.95198E-04 0.01940  5.81197E-04 0.02178  1.40546E-03 0.01418  3.18324E-04 0.03031  7.05204E-05 0.06239 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.08703E-01 0.02447  1.24864E-02 0.00038  3.22025E-02 0.00027  1.06804E-01 0.00102  3.01214E-01 0.00054  1.25426E+00 0.00182  6.78051E+00 0.03140 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.43302E-07 0.01974  7.42908E-07 0.01974  8.59125E-07 0.05213 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.72904E-07 0.01974  6.72549E-07 0.01973  7.77673E-07 0.05211 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01601E-03 0.04113  1.80739E-04 0.14345  7.37835E-04 0.08661  5.60976E-04 0.09776  1.21688E-03 0.05951  2.35530E-04 0.12954  8.40573E-05 0.21625 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.94220E-01 0.12351  1.24788E-02 9.9E-05  3.22612E-02 0.00096  1.07146E-01 0.00362  3.01537E-01 0.00192  1.25992E+00 0.00680  7.16318E+00 0.08160 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98222E-03 0.04127  1.81207E-04 0.14435  7.29808E-04 0.08362  5.68429E-04 0.09539  1.18650E-03 0.05960  2.29986E-04 0.12921  8.62906E-05 0.21435 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.98283E-01 0.12513  1.24788E-02 9.9E-05  3.22579E-02 0.00093  1.07155E-01 0.00362  3.01478E-01 0.00191  1.25966E+00 0.00684  7.16318E+00 0.08160 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.07382E+03 0.03581 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.40942E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.61064E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.36302E-03 0.00547 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.99888E+03 0.00537 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.14175E-09 0.00101 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.94101E-05 0.00512  6.93459E-05 0.00514  5.06542E-06 0.22536 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.02597E-05 0.01814  6.02482E-05 0.01820  4.06611E-06 0.30528 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.14118E-04 0.01761  3.13344E-04 0.01761  5.18844E-04 0.21727 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59663E+01 0.01195 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46697E+01 0.00027  3.86672E+01 0.00039 ];

