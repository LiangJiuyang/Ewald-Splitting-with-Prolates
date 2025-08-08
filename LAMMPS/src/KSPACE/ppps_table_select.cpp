#include "ppps_timing.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "grid3d.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "remap_wrap.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;


/* ---------------------------------------------------------------------- 
   Build Table
------------------------------------------------------------------------- */

int PPPSTiming::build_table(double algorithm_accuracy, double spreading_accuracy)
{
    double options[12] = {0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002, 0.000001};
    int options_size = 12;
    int options_smooth = 0;//1;
    
    double closet = options[0];
    double min_diff = std::fabs(options[0] - algorithm_accuracy);
    for (int i = 1; i < options_size; ++i) {
        double diff = std::fabs(options[i] - algorithm_accuracy);
        if (diff < min_diff) {
            min_diff = diff;
            closet = options[i];
        }
    }
    
    if(me==0)
      printf("The selected relative error level is %lf\n", closet);
    
    if(closet == 0.005)
    {
      // C = 7.7625    Lambda_0 = 0.89968068782467;
      select_c = 7.7625;
      Lambda_0 = 0.89968068782467;
      num_of_force_poly = 10;
      num_of_energy_poly = 10;
      num_of_Fourier_poly = 9;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={0.999999324024297,	0.000101746872439440,	-0.00201403463508854,	-5.16990722741147,	0.144801668581986,	8.06455061612519,	5.59289884198272,	-21.3916361943072,	15.3766233153695,	-3.60641020760809};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999889284700,	-2.22298548116544,	-0.000970540289881722,	2.60166723328620,	-0.110346381304243,	-1.87128628432648,	-1.32927266978829,	3.75049722774856,	-2.28662667327734,	0.469323728820715};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={1.99999575614539,	0.000802359081806120,	-7.00810989059276,	0.305442067393477,	8.73766808567801,	6.54360599405041,	-22.9100580526287,	16.0356686147453,	-3.69691074016562};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.002)
    {
      // C = 8.7826    Lambda_0 = 0.845820643995472;
      select_c = 8.7826;
      Lambda_0 = 0.845820643995472;
      num_of_force_poly = 12;
      num_of_energy_poly = 10;
      num_of_Fourier_poly = 11;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={1.00000016792146,	-4.63616027092731e-05,	0.00209482327769395,	-6.34771368826480,	0.320102182467991,	11.8330502754911,	4.67961570340904,	-22.8704843569304,	5.08944696550660,	16.0849167644219,	-12.6618273113199,	2.87453777038353};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000010540748,	-2.36458085533627,	0.000130278845119097,	3.15888221592848,	-0.0616284058308989,	-2.93238259422184,	-1.54048852720144,	5.75029278906965,	-3.87540469613228,	0.865180132545413};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={1.99999971027609,	6.25661289735818e-05,	-8.00940582544042,	0.0263890770367521,	14.0534302009323,	0.249235430292060,	-14.0861358062802,	-5.37333828709231,	23.0664570779660,	-15.2061033775976,	3.28253317835816};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.001)
    {
       // C = 9.539199999999999  Lambda_0 = 0.811584854067189;
       if(options_smooth == 0) {
        select_c = 9.539199999999999;
        Lambda_0 = 0.811584854067189;
        num_of_force_poly = 12;
        num_of_energy_poly = 11;
        num_of_Fourier_poly = 12; 
        memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
        double poly_coeff_f[]={1.00000039989940,	-0.000111962487259587,	0.00514843639711582,	-7.29264165850299,	0.830181749158404,	12.6186620024320,	13.6708987736821,	-47.5287345877297,	27.7974058124898,	8.99141286159203,	-13.7437099200624,	3.65337584091615};
        for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
        double poly_coeff_e[]={0.999999716772465,	-2.46424478449351,	-0.00280410611862911,	3.64468157293033,	-0.356765988541030,	-2.56379487753728,	-4.83246201861727,	12.3138189974739,	-9.64688979608662,	3.33144581696123,	-0.422984308746543};
        for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
        double poly_coeff_Fourier[]={2.00000025719292,	-7.52257923832891e-05,	-8.76261588292675,	-0.0692064563534369,	17.8657483926914,	-3.92467980743501,	-5.82189850092092,	-34.1528521513068,	69.1349621821723,	-51.8399132143098,	17.9813979062204,	-2.40933563352913};
        for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
      }
      else if(options_smooth==1)
      {
        select_c = 9.539199999999999;
        num_of_force_poly = 11;
        num_of_energy_poly = 11;
        num_of_Fourier_poly = 10;
        memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
        double poly_coeff_f[] = {0.999999716772465,	0,	0.00280410611862911,	-7.28936314586065,	1.07029796562309,	10.2551795101491,	24.1623100930863,	-73.8829139848432,	67.5282285726064,	-26.6368697944564,	3.79032696080417};
        for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
        double poly_coeff_e[] = {0.999999716772465,	-2.46424478449351,	-0.00280410611862911,	3.64468157293033,	-0.356765988541030,	-2.56379487753728,	-4.83246201861727,	12.3138189974739,	-9.64688979608662,	3.32960872430705,	-0.421147440089352};
        for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
        double poly_coeff_Fourier[] = {1.99998955589830,	0.00199835972837489,	-8.83039128509984,	0.860992427259938,	11.5382890937144,	20.8598769738192,	-66.1074850343676,	59.4032412429110,	-22.9037695062821,	3.17897880395466};
        for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
      }
    }
    else if(closet == 0.0005)
    {
      // C = 10.29    Lambda_0 = 0.781415895482355;
      select_c = 10.29;
      Lambda_0 = 0.781415895482355;
      num_of_force_poly = 13;
      num_of_energy_poly = 12;
      num_of_Fourier_poly = 12;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={0.999999937934452,	2.22000938196812e-05,	-0.00132056539060987,	-8.09018838700313,	-0.379170743154443,	23.6968164549766,	-13.0465638135469,	12.1757149133300,	-88.7453494169642,	155.911091312689,	-121.935353387077,	46.5579043051199,	-7.14263973476524};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999944619443,	-2.55944108955882,	-0.000704071998158612,	4.07301084968056,	-0.111059945527381,	-4.66367646280159,	-1.74084972953546,	7.98720721513622,	-2.83188665169417,	-3.54373240959362,	3.11566696906769,	-0.724534687883988};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000036621317,	-0.000108381114053063,	-9.51369860946367,	-0.102484724095619,	21.4604858052176,	-6.01434548122842,	-4.03871154322870,	-54.6821495653678,	110.755147242626,	-88.0409626509975,	33.0900384636087,	-4.91245815005829};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.0002)
    {
      // C = 11.284    Lambda_0 = 0.74620541222595;
      select_c = 11.284;
      Lambda_0 = 0.74620541222595;
      num_of_force_poly = 13;
      num_of_energy_poly = 13;
      num_of_Fourier_poly = 13;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={0.999999970324060,	1.29484638151656e-05,	-0.000912831960741989,	-9.36946746428633,	-0.347475436385835,	29.8590128346239,	-15.0698899852300,	10.6121783341436,	-125.049482171426,	243.235626523391,	-204.860047126553,	83.6981139163635,	-13.7072777555374};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000001182555,	-2.68023134366658,	0.000248144151621776,	4.69138061023308,	0.0702465528551177,	-7.25813025555189,	2.38283879498389,	-0.450933220187718,	15.9816929265746,	-28.5957587707567,	21.6452792486930,	-7.96945073468913,	1.18281803202510};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000016490129,	-5.50364661243627e-05,	-10.5121103377570,	-0.0652744501382609,	25.9024145489325,	-4.73130800931842,	-17.1886271203800,	-52.1798425235686,	127.321523257224,	-96.7939209351428,	24.0170537274156,	4.72219612307015,	-2.49175719330991};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if (closet == 0.0001)
    {
      // C = 1.202400000000000e+01  Lambda_0 = 7.228787365921187e-01;
      if(options_smooth == 0) {
      select_c = 1.202400000000000e+01;
      Lambda_0 = 7.228787365921187e-01;
      num_of_force_poly = 13;
      num_of_energy_poly = 13;
      num_of_Fourier_poly = 12; 
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={1.00000006021466,	-1.54317704013085e-05,	0.000544031060444539,	-10.3848656437997,	-0.0685913949496211,	33.4873483886740,	-10.6190151594428,	-7.00864746526383,	-132.636474376189,	301.342471560251,	-271.952309118610,	116.792012347112,	-19.9522584995319};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000001075386,	-2.76671977587223,	0.000248133153658028,	5.18451117975703,	0.0771836301689182,	-8.61808700165139,	2.87598455859069,	-0.402380991626970,	21.1916233438731,	-39.8233819500516,	31.5480841091314,	-12.1563446129808,	1.88927937006230};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000042108038,	-0.000134647049399250,	-11.2493087115706,	-0.149333841730986,	30.6437784771564,	-10.3623431238639,	-3.62591877740590,	-112.606453705167,	247.704783925638,	-219.879217495585,	93.3181417425103,	-15.7938501637491};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
      }
      else if(options_smooth==1)
      { 
        select_c = 1.202400000000000e+01;
        num_of_force_poly = 13;
        num_of_energy_poly = 13;
        num_of_Fourier_poly = 14;
        memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
        double poly_coeff_f[] = {1.00000001075386,	0,	-0.000248133153724128,	-10.3690223595126,	-0.231550890509191,	34.4723480064839,	-14.3799227918144,	2.41428594500155,	-148.341363395835,	318.587055584321,	-283.932756968469,	121.565427099596,	-20.7842521068624};
        for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
        double poly_coeff_e[] = {1.00000001075386,	-2.76671977587224,	0.000248133153724128,	5.18451117975632,	0.0771836301697304,	-8.61808700162097,	2.87598455836288,	-0.402380990833592,	21.1916233422621,	-39.8233819480402,	31.5480841076077,	-12.1565427099596,	1.88927936994588};
        for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
        double poly_coeff_Fourier[] = {2.00000001790959,	1.88498352825597e-06,	-11.2563035266662,	0.00507365739771171,	28.9373463013601,	0.818760925206846,	-50.8967155380915,	21.7735908093941,	-14.1154609032919,	130.535730073169,	-223.506116022127,	169.169388716040,	-62.9568350856495,	9.49166702541675};
        for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
      }
    }
    else if(closet == 0.00005)
    {
      // C = 12.762    Lambda_0 = 0.701666211927114;
      select_c = 12.762;
      Lambda_0 = 0.701666211927114;
      num_of_force_poly = 15;
      num_of_energy_poly = 13;
      num_of_Fourier_poly = 14;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={0.999999986607656,	5.76073596343998e-06,	-0.000407096211643088,	-11.3860527546485,	-0.158349572595723,	39.0930144922175,	-6.77647358124247,	-46.4914032771385,	-44.1702401277728,	122.205435569987,	43.1836489722343,	-251.443029879477,	237.071110445860,	-97.8950384518770,	15.7678809123309};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000000419986,	-2.85036017962769,	0.000154088936279329,	5.69421498608091,	0.0640859096827109,	-9.99055823953118,	2.93923507538692,	0.947222912635596,	25.4126861600602,	-51.7646800240185,	43.0408804276299,	-17.2841886975908,	2.79130759367482};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000001194615,	-3.86571327920526e-06,	-11.9954121028514,	-0.00304217521748074,	33.1503178228963,	0.171799605540627,	-58.8267723371113,	15.8876689652676,	7.05752168407048,	143.029491643709,	-285.922560575012,	231.896117971262,	-90.7097165432157,	14.2646610657477};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.00002)
    {
      // C = 13.74    Lambda_0 = 0.676233322943516;
      select_c = 13.74;
      Lambda_0 = 0.676233322943516;
      num_of_force_poly = 16;
      num_of_energy_poly = 15;
      num_of_Fourier_poly = 15;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={1.00000000567200,	-2.98196419497992e-06,	0.000260640145792745,	-12.8003649057351,	0.164047500990804,	44.3391527234620,	13.0549333308177,	-157.067309245327,	231.475384345079,	-471.547765905995,	1109.43460723270,	-1602.07064591235,	1357.65431622232,	-679.093556142915,	187.745901912535,	-22.2889177054347};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000000153754,	-2.95755972859943,	4.57485992839748e-05,	6.39443462352736,	0.0171275345196431,	-11.6734683047598,	0.671843256336931,	13.3305469326335,	2.96440161111971,	-14.2797616079681,	-15.3559467546313,	46.0799937811758,	-39.0566807487586,	15.2199191798773,	-2.35489552708168};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={1.99999997707086,	1.04451716529015e-05,	-12.9756922246350,	0.0235625628625131,	38.6382080817105,	3.45535769454941,	-93.6376209948496,	88.0985023142841,	-163.470572649550,	531.433710715431,	-858.270211226228,	745.653364549636,	-367.758084993675,	97.6747644619343,	-10.8652708965088};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.00001)
    {
      // C = 14.471    Lambda_0 = 0.658932096384228;
      select_c = 14.471;
      Lambda_0 = 0.658932096384228;
      num_of_force_poly = 16;
      num_of_energy_poly = 15;
      num_of_Fourier_poly = 16;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={1.00000000627485,	-3.36432782963719e-06,	0.000300123003584007,	-13.8782436261756,	0.197257524631802,	50.8388120388574,	16.4529275347297,	-196.722236945548,	307.041042891568,	-649.608186153088,	1556.39999914506,	-2311.07135701978,	2026.63301561114,	-1052.86835501504,	303.165930162284,	-37.5808820459041};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000000326453,	-3.03521553166402,	0.000101136693413356,	6.93099908700899,	0.0405355301947390,	-13.6085983351212,	1.83580148237586,	12.3830231388235,	14.1235805433740,	-37.8920251280739,	4.11573851197565,	42.5007094799800,	-43.9541501704656,	18.5819372076673,	-3.02243696048009};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={1.99999999439007,	2.80903979866208e-06,	-13.706978960208,	0.00755907218940433,	43.5764468221711,	1.29771317687383,	-95.0103365330133,	37.6251651279966,	4.88313386315588,	240.885110401028,	-456.714217553043,	259.638979565463,	77.4174111184462,	-168.302812795116,	79.7511566652626,	-13.3483190317671};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.000005)
    {
      // C = 15.2    Lambda_0 = 0.642936586623448;
      select_c = 15.2;
      Lambda_0 = 0.642936586623448;
      num_of_force_poly = 18;
      num_of_energy_poly = 16;
      num_of_Fourier_poly = 17;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={0.999999999961415,	3.50514406510100e-08,	-4.97213805364383e-06,	-14.9690649467868,	-0.00747690008593338,	60.6665266966819,	-1.34644856393226,	-126.098008769228,	-54.3352122865931,	423.232596807320,	-636.428142801519,	1177.79162790563,	-2341.66508961917,	3051.02310311967,	-2426.06101122707,	1163.65967928197,	-313.041710575931,	36.5786473990644};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999999099120,	-3.11072621473413,	-4.16989140530205e-05,	7.48611554575215,	-0.0264660335322304,	-14.8428514991661,	-2.12675694320169,	33.3704991963947,	-38.1390668763677,	72.6529943944285,	-185.189878437022,	272.287233986797,	-229.644249207414,	113.803259994199,	-31.2040207619571,	3.68395455524089};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000000086881,	-5.25126781296750e-07,	-14.4364458414452,	-0.00210846571103307,	48.7005141353567,	-0.568169260557326,	-97.3525740598455,	-28.2798553232208,	270.277232059089,	-372.867554665144,	694.096147733748,	-1467.57956864527,	1964.25339236937,	-1574.79743149223,	755.550258489819,	-202.519172716831,	23.5253430109665};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.000002)
    {
      // C = 16.169    Lambda_0 = 0.623373525060067;
      select_c = 16.169;
      Lambda_0 = 0.623373525060067;
      num_of_force_poly = 18;
      num_of_energy_poly = 16;
      num_of_Fourier_poly = 17;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={1.00000000034339,	-2.02976311321212e-07,	1.94981453826570e-05,	-16.4770715625018,	0.0133019096442411,	71.2794699277118,	0.785151753393450,	-174.531205061627,	-9.95656047618832,	387.067796455747,	-457.820058094369,	967.589109031502,	-2618.02073256696,	3978.91837173825,	-3455.94034979385,	1764.11522775444,	-498.878676874510,	60.8562108409938};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999999045266,	-3.20834876144458,	-4.67857108627234e-05,	8.23985059888790,	-0.0315364189413687,	-17.4942968522807,	-2.70077010425988,	42.8255756352583,	-51.8167938635987,	103.197300670063,	-270.363676236796,	412.169470341688,	-362.153605212915,	187.496958597706,	-53.8273657694752,	6.66728416275876};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000000057108,	-3.78164314113194e-07,	-15.4063416444021,	-0.00179127670139767,	55.6899596248698,	-0.559248413241051,	-120.675801465614,	-31.8086925550881,	343.550217507418,	-474.106275313955,	919.463046661886,	-2091.24657508016,	2964.02976513131,	-2495.76993831685,	1253.35441671873,	-351.100386357669,	42.5876478236263};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.000001) // here splitting arruracy/1000 = poly appr accuray
    {
      // C = 16.894    Lambda_0 = 0.6098509283417469;
      select_c = 16.894;
      Lambda_0 = 0.6098509283417469; 
      num_of_force_poly = 19;
      num_of_energy_poly = 18;
      num_of_Fourier_poly = 18;
      memory->create(force_poly_coeff, num_of_force_poly, "ppps:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "ppps:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "ppps:force_poly_coeff");
      double poly_coeff_f[]={0.999999999638727,	2.63479773256192e-07,	-3.19417418848389e-05,	-17.6333443474892,	-0.0390074317994884,	80.8707730961580,	-6.20070503290708,	-160.018057923742,	-232.914645924401,	1260.88466321026,	-2648.13558893027,	5420.17767331231,	-9881.03096591064,	12839.8625899721,	-11174.3286120305,	6403.93704775573,	-2331.23540176163,	490.206596129121,	-45.4029803378526};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999999991520,	-3.27948996189663,	-1.80568509981077e-07,	8.81743754190410,	0.000320571611515774,	-20.0751581733884,	0.117573204335827,	33.0624961311052,	6.30622387244858,	-71.9042114249266,	86.8411829564886,	-158.152192190952,	355.697628537825,	-489.373546883985,	397.550384665017,	-192.665498983712,	52.1956206204255,	-6.13877030177604};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000000107776,	-6.99610317692924e-07,	-16.1318968313873,	-0.00321724639537002,	61.2608830862544,	-0.986114111815940,	-136.704994512668,	-55.6112898651284,	495.302055078790,	-830.514709740437,	1732.34815539406,	-3724.51825917900,	5274.81626767449,	-4667.05849842013,	2589.09283785930,	-870.921259853291,	159.121743274757,	-11.4917005872599};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }

    double spreading_options[8] = {0.005,  0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001};
    int spreading_options_size = 8;
    int spreading_options_smooth = 0;//1;
    double spreading_closet = spreading_options[0];
    double spreading_min_diff = std::fabs(spreading_options[0] - spreading_accuracy);
    for (int i = 1; i < spreading_options_size; ++i) {
        double spreading_diff = std::fabs(spreading_options[i] - spreading_accuracy);
        if (spreading_diff < spreading_min_diff) {
            spreading_min_diff = spreading_diff;
            spreading_closet = spreading_options[i];
        }
    }
    
    if(me==0)
      printf("The selected spreading accuracy level is %.9g\n", spreading_closet);
    
    if(spreading_closet == 0.000001){
        spreading_select_c = 16.894;
        spreading_Lambda_0 = 0.609850928341748;
 
        Fourier_spreading_order = 13;
        
          order = 8;
          poly_order = 9;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
          
          double array[9][8] = {
            {  0.000592989202348864,   0.0633126581878211,    0.660888450918412,    1.88656089945326,    1.88656089945326,    0.660888450918412,    0.0633126581878211,    0.000592989202348864},
            {  0.00414755568114888,   0.202181904719444,    1.07482223510981,    0.958256993886814,   -0.958256993886814,   -1.07482223510981,   -0.202181904719444,   -0.00414755568114888},
            {  0.0121303864875324,   0.257764148025186,    0.459597366701701,   -0.729492335870010,   -0.729492335870010,    0.459597366701701,    0.257764148025186,    0.0121303864875324},
            {  0.0192882427467056,   0.151807064333858,   -0.157030799069369,   -0.422700847287472,    0.422700847287472,    0.157030799069369,   -0.151807064333858,   -0.0192882427467056},
            {  0.0178956972698489,   0.0203142848700180,   -0.167761301713827,    0.129588044705475,    0.129588044705475,   -0.167761301713827,    0.0203142848700180,    0.0178956972698489},
            {  0.00920764795075203,  -0.0241143299623000,   -0.00877555713801762,    0.0835828678370907,   -0.0835828678370907,    0.00877555713801762,    0.0241143299623000,   -0.00920764795075203},
            {  0.00156673745411895,  -0.0114382770658698,    0.0232903569374928,   -0.0134666600221579,   -0.0134666600221579,    0.0232903569374928,   -0.0114382770658698,    0.00156673745411895},
            { -0.000969198819341558,   0,                     0,                     0,                     0,                     0,                     0,                     0.000969198819341558},
            { -0.000552572940062616,   0,                     0,                     0,                     0,                     0,                     0,                    -0.000552572940062616}
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
        double Fourier_array[13] = {1.30567848375104,	-0.000338881382634854,	-10.5114320206880,	-0.471441078279729,	45.6847889816358,	-41.5425712922094,	97.2327610888870,	-590.153065248936,	1370.43842760087,	-1615.37545487527,	1063.58996875185,	-376.425801197314,	56.2284803885425};
        for(int i=0; i<Fourier_spreading_order; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        } 
    }
    else if(spreading_closet == 0.000005){
       spreading_select_c = 15.2;
       spreading_Lambda_0 = 0.642936586623448;
        
       Fourier_spreading_order = 12;

       order = 7;
        poly_order = 8;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
        double array[8][7] = {
            {  0.00212433200815181,   0.158300082442380,    1.14218066027196,    2.08370682348781,    1.14218066027196,    0.158300082442380,    0.00212433200815181},
            {  0.0139927145692719,   0.450271276939759,    1.40185037879945,    2.00000000000000e-17,   -1.40185037879945,   -0.450271276939759,   -0.0139927145692719},
            {  0.0378372092878287,   0.476137399214982,    0.0999425846138216,   -1.22778768744038,    0.0999425846138216,    0.476137399214982,    0.0378372092878287},
            {  0.0539813571819290,   0.177497160705514,   -0.516286122380645,    8.00000000000000e-17,    0.516286122380645,   -0.177497160705514,   -0.0539813571819290},
            {  0.0422641501971563,  -0.0556215631142325,   -0.155097441228934,    0.337288188324941,   -0.155097441228934,   -0.0556215631142325,    0.0422641501971563},
            {  0.0147701103486458,  -0.0624319364762037,    0.0817026327008929,    3.20000000000000e-16,   -0.0817026327008929,    0.0624319364762037,   -0.0147701103486458},
            { -0.00266873539811322,  -0.00804165064544980,    0.0367620701997351,   -0.0545257114731151,    0.0367620701997351,   -0.00804165064544980,   -0.00266873539811322},
            { -0.00374582804565569,   0,                     0,                     0,                     0,                     0,                     0.00374582804565569}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
       double Fourier_array[12] = {1.33968944061042,	0.000543509123502033,	-9.69355941938164,	0.376610762608276,	29.6645680323095,	11.6432241675566,	-87.6674548817399,	-22.1097740762224,	266.829651615066,	-324.062468086942,	166.389618396144,	-32.7106481571401};
       for(int i=0; i<12; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.00001){
        spreading_select_c = 14.471;
        spreading_Lambda_0 = 0.658932096384228;

        Fourier_spreading_order = 12;
        order = 7;
          poly_order = 8;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
          double array[8][7] = {
              {  0.00298919664169506,   0.178171528579562,    1.16266914545820,    2.05753047598323,    1.16266914545820,    0.178171528579562,    0.00298919664169506},
              {  0.0186495595974477,   0.480917478086805,    1.35471690638510,    2.00000000000000e-17,   -1.35471690638510,   -0.480917478086805,   -0.0186495595974477},
              {  0.0472324139989913,   0.473930990088334,    0.0546323377224630,   -1.15108367882841,    0.0546323377224630,    0.473930990088334,    0.0472324139989913},
              {  0.0620916592785675,   0.150891764895235,   -0.486836401682256,    8.00000000000000e-17,    0.486836401682256,   -0.150891764895235,   -0.0620916592785675},
              {  0.0433315645968559,  -0.0676652865968097,   -0.126778398614065,    0.299208924834946,   -0.126778398614065,   -0.0676652865968097,    0.0433315645968559},
              {  0.0117239007564396,  -0.0563717567866057,    0.0761025698637772,    3.20000000000000e-16,   -0.0761025698637772,    0.0563717567866057,   -0.0117239007564396},
              { -0.00425730074812564,   0,                     0.0294797809953460,   -0.0457588204974441,    0.0294797809953460,    0,                    -0.00425730074812564},
              { -0.00355428349991723,   0,                     0,                     0,                     0,                     0,                     0.00355428349991723}
            };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
        double Fourier_array[12] = {1.35577192368550,	0.000268246369062156,	-9.30243836770539,	0.157630619614275,	28.6282226519485,	2.03270243653437,	-49.7425756717489,	-68.3449372595392,	277.808550834175,	-301.711506508906,	147.042555727368,	-27.9242374221768};
        for(int i=0; i<12; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        }        
    }
    else if(spreading_closet == 0.00005){
      spreading_select_c = 12.762;
      spreading_Lambda_0 = 0.701666211927114;  
      Fourier_spreading_order = 12;
      order = 6;
      poly_order = 7;
      memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
      memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
      double array[7][6] = {
          {  0.0102240357539297,   0.401889526927379,    1.68432250504309,    1.68432250504309,    0.401889526927379,    0.0102240357539297},
          {  0.0590052369375671,   0.920034892194361,    1.13747495459263,   -1.13747495459263,   -0.920034892194361,   -0.0590052369375671},
          {  0.134307754290813,    0.649387020557088,   -0.783484341450943,  -0.783484341450943,    0.649387020557088,    0.134307754290813},
          {  0.150254949492937,   -0.0344564275058532,  -0.637968124286612,    0.637968124286612,    0.0344564275058532,   -0.150254949492937},
          {  0.0751132858897679,  -0.224306795157137,    0.153301553803058,    0.153301553803058,   -0.224306795157137,    0.0751132858897679},
          { -0.00532068674755281, -0.0571238945201862,    0.156018426061326,   -0.156018426061326,    0.0571238945201862,    0.00532068674755281},
          { -0.0196638111454918,   0,                     0,                     0,                     0,                    -0.0196638111454918}
      };
      for(int i=0; i<poly_order; i++){
          for(int j=0; j<order; j++){
            rho_coeff[i][j+(1-order)/2] = array[i][j];
          }
      }
       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
       double Fourier_array[12] = {1.33968944061042,	0.000543509123502033,	-9.69355941938164,	0.376610762608276,	29.6645680323095,	11.6432241675566,	-87.6674548817399,	-22.1097740762224,	266.829651615066,	-324.062468086942,	166.389618396144,	-32.7106481571401};
       for(int i=0; i<12; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.0001)
    {   
        spreading_select_c = 1.202400000000000e+01;
        spreading_Lambda_0 = 7.228787365921187e-01;
        
        Fourier_spreading_order = 10;
        
        if (order == 5){
          poly_order = 7;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
          double array[7][5] = {
              {  0.0233335706470471,   0.768290073124929,    1.96132013624477,    0.768290073124929,    0.0233335706470471},
              {  0.134808438344117,    1.50167357405590,    2.00000000000000e-17,   -1.50167357405590,   -0.134808438344117},
              {  0.301631728014790,    0.578139762427498,   -1.76023178613503,    0.578139762427498,    0.301631728014790},
              {  0.314515337115647,   -0.615018072688094,    8.00000000000000e-17,    0.615018072688094,   -0.314515337115647},
              {  0.112912662468230,   -0.428484936729818,    0.663158589174627,   -0.428484936729818,    0.112912662468230},
              { -0.0638319824884254,    0.0819255603277930,    0,   -0.0819255603277930,    0.0638319824884254},
              { -0.0597832562237622,    0,                     0,                     0,                    -0.0597832562237622}
            };
            for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
            }
        }
        else if (order == 6){
          poly_order = 7;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
          double array[7][6] = {
              {  0.0140173259281708,   0.436940980869284,    1.67578083257896,    1.67578083257896,    0.436940980869284,    0.0140173259281708},
              {  0.0756793887491835,   0.938090058260092,    1.06192364350664,   -1.06192364350664,   -0.938090058260092,   -0.0756793887491835},
              {  0.158396902138095,    0.595914995080751,   -0.753423532492532,  -0.753423532492532,    0.595914995080751,    0.158396902138095},
              {  0.158046145356951,   -0.0746561516123935,  -0.560222777140656,    0.560222777140656,    0.0746561516123935,   -0.158046145356951},
              {  0.0638507799974509,  -0.207178497762158,    0.144482927492809,    0.144482927492809,   -0.207178497762158,    0.0638507799974509},
              { -0.0135904695582716,  -0.0387276123244048,    0.128796509230334,   -0.128796509230334,    0.0387276123244048,    0.0135904695582716},
              { -0.0189112616571613,   0,                     0,                     0,                     0,                    -0.0189112616571613}
            };
            for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
            }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
        double Fourier_array[10] = {1.41784303282410,	0.00294142013361292,	-8.08421557278830,	1.43738429991616,	10.4837622284399,	40.6381801567710,	-130.779969354531,	140.921778932332,	-69.2859039528575,	13.2482975813580};
        for(int i=0; i<10; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        }  
    }
    else if(spreading_closet == 0.0005){
       spreading_select_c = 10.29;
       spreading_Lambda_0 = 0.781415895482355;
        
       Fourier_spreading_order = 7;

       order = 5;
       poly_order = 6;
       memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
       memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
       double array[6][5] = {
           {  0.0449230045055491,   0.852935389717147,    1.88347600828701,    0.852935389717147,    0.0449230045055491},
           {  0.217859419942785,    1.40649181234108,    2.00000000000000e-17,   -1.40649181234108,   -0.217859419942785},
           {  0.389194231489898,    0.332050124324953,   -1.43094596318676,    0.332050124324953,    0.389194231489898},
           {  0.286229438378169,   -0.550455522494631,    8.00000000000000e-17,    0.550455522494631,   -0.286229438378169},
           {  0.00990332722501597,  -0.262638725203169,    0.456021360004942,   -0.262638725203169,    0.00990332722501597},
           { -0.0829796301229596,    0,                     0,                     0,                     0.0829796301229596}
       };
       for(int i=0; i<poly_order; i++){
           for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
           }
       }
       
       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
       double Fourier_array[7] = {1.47167640468885,	0.00508436012575368,	-6.90622739527608,	-2.06454565711960,	26.7380877688229,	-28.5427551373625,	9.29983535407476};
       for(int i=0; i<7; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.001){
       spreading_select_c = 9.5392;
       spreading_Lambda_0 = 0.811584854067189;
        
       Fourier_spreading_order = 7;

       if(order == 4){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
        rho_coeff[0][-1] = 0.099693018874845; rho_coeff[1][-1] = 0.476781923984876; rho_coeff[2][-1] = 0.800880039422003; rho_coeff[3][-1] = 0.450560633317604; rho_coeff[4][-1] = -0.192702230561618; rho_coeff[5][-1] = -0.264856686543029;
        rho_coeff[0][0] = 1.39835440671411; rho_coeff[1][0] = 1.57840096107687; rho_coeff[2][0] = -0.784972435014229; rho_coeff[3][0] = -1.34359649381689; rho_coeff[4][0] = 0.132862727353174; rho_coeff[5][0] = 0.476301673064559;
        rho_coeff[0][1] = 1.39835440671411; rho_coeff[1][1] = -1.57840096107687; rho_coeff[2][1] = -0.784972435014229; rho_coeff[3][1] = 1.34359649381689; rho_coeff[4][1] = 0.132862727353174; rho_coeff[5][1] = -0.476301673064559;
        rho_coeff[0][2] = 0.099693018874845; rho_coeff[1][2] = -0.476781923984876; rho_coeff[2][2] = 0.800880039422003; rho_coeff[3][2] = -0.450560633317604; rho_coeff[4][2] = -0.192702230561618; rho_coeff[5][2] = 0.264856686543029;
       }

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
       Fourier_spreading_coeff[0] = 1.498660780149550; Fourier_spreading_coeff[1] = -9.796546007810547e-03; Fourier_spreading_coeff[2] = -6.279938459904613e+00; Fourier_spreading_coeff[3] = -2.786353502254686e+00; Fourier_spreading_coeff[4] = 2.487895712110857e+01; Fourier_spreading_coeff[5] = -2.520382479398788e+01; Fourier_spreading_coeff[6] = 7.903787290246450;
    }
    else if(spreading_closet == 0.005){
       spreading_select_c = 7.7625;
       spreading_Lambda_0 = 0.89968068782467;
        
       Fourier_spreading_order = 6;

       order = 3;
       poly_order = 5;
       memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"pppm:rho_coeff");
       memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"pppm:drho_coeff");
       double array[5][3] = {
            {  0.303558251056439,   1.74838160615300,    0.303558251056439},
            {  1.22754476618614,    2.00000000000000e-17,   -1.22754476618614},
            {  1.41111797811199,   -2.69060102802187,    1.41111797811199},
            { -0.237031733904531,    8.00000000000000e-17,    0.237031733904531},
            { -1.05339856461695,    1.58177644858677,   -1.05339856461695}
       };
       for(int i=0; i<poly_order; i++){
           for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
           }
       }
    memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "ppps:Fourier_spreading_coeff");
    double Fourier_array[6] = {1.57136596057693,	0.129296803579999,	-6.85700115006353,	4.75912196700269,	3.02225987653029,	-2.62080742890787};
    for(int i=0; i<6; i++){
        Fourier_spreading_coeff[i] = Fourier_array[i];
    } 
    }

    for (int m = -(order-1)/2; m <= order/2; m += 1) {
        for (int l = 1; l < poly_order; l++)
            drho_coeff[l-1][m] = l*rho_coeff[l][m]; // Coefficients for l x^l-1 terms
        drho_coeff[poly_order-1][m] = 0.00;    
    }
    return 0;
}
