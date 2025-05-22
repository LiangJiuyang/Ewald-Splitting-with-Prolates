#include "esp.h"

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

int ESP::build_table(double algorithm_accuracy, double spreading_accuracy)
{
    double options[16] = {0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002, 0.000001, 0.0000005, 0.0000002, 0.0000001};
    int options_size = 16;
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
    
    if(closet == 0.01)
    {
       // C = 6.9862  Lambda_0 = 0.948344618546107;
        select_c = 6.9862;
        Lambda_0 = 0.948344618546107;
        num_of_force_poly = 8;
        num_of_energy_poly = 7;
        num_of_Fourier_poly = 7; 
        memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
        double poly_coeff_f[]={0.999976976970184,	0.00346624698329073,	-0.0862705381209952,	-3.53479050138590,	-3.91423942767437,	17.0946354870164,	-14.4017862335112,	3.85655992310600};
        for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
        double poly_coeff_e[]={0.999939779902725,	-2.10332973887343,	-0.0829743671199074,	2.61648482694035,	-0.962766737007719,	-1.06417462979906,	0.596893130500942};
        for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
        double poly_coeff_Fourier[]={2.00024199443689,	-0.0252958033638162,	-5.76679407926988,	-2.83817635377886,	17.0964398708927,	-14.0232785626624,	3.57338390897574};
        for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.005)
    {
      // C = 7.7625    Lambda_0 = 0.89968068782467;
      select_c = 7.7625;
      Lambda_0 = 0.89968068782467;
      num_of_force_poly = 10;
      num_of_energy_poly = 10;
      num_of_Fourier_poly = 9;
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
        memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
        memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
        memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
        memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
        memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
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
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
      double poly_coeff_f[]={0.999999999638727,	2.63479773256192e-07,	-3.19417418848389e-05,	-17.6333443474892,	-0.0390074317994884,	80.8707730961580,	-6.20070503290708,	-160.018057923742,	-232.914645924401,	1260.88466321026,	-2648.13558893027,	5420.17767331231,	-9881.03096591064,	12839.8625899721,	-11174.3286120305,	6403.93704775573,	-2331.23540176163,	490.206596129121,	-45.4029803378526};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999999991520,	-3.27948996189663,	-1.80568509981077e-07,	8.81743754190410,	0.000320571611515774,	-20.0751581733884,	0.117573204335827,	33.0624961311052,	6.30622387244858,	-71.9042114249266,	86.8411829564886,	-158.152192190952,	355.697628537825,	-489.373546883985,	397.550384665017,	-192.665498983712,	52.1956206204255,	-6.13877030177604};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000000107776,	-6.99610317692924e-07,	-16.1318968313873,	-0.00321724639537002,	61.2608830862544,	-0.986114111815940,	-136.704994512668,	-55.6112898651284,	495.302055078790,	-830.514709740437,	1732.34815539406,	-3724.51825917900,	5274.81626767449,	-4667.05849842013,	2589.09283785930,	-870.921259853291,	159.121743274757,	-11.4917005872599};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.0000005)
    {
      // C = 17.617    Lambda_0 = 0.597205717649591;
      select_c = 17.617;
      Lambda_0 = 0.597205717649591;
      num_of_force_poly = 20;
      num_of_energy_poly = 18;
      num_of_Fourier_poly = 19;
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
      double poly_coeff_f[]={0.999999999907053,	7.30236850987609e-08,	-9.51015362621321e-06,	-18.8154799642815,	-0.0132912073605151,	89.9360838477049,	-2.38739557034506,	-222.056934960936,	-99.5130494520633,	843.299583785857,	-1214.49369126173,	2124.42025826931,	-4477.21418852592,	5752.07408805164,	-3452.47857543231,	-264.533604482836,	1890.09514360170,	-1309.11506830294,	412.117231418881,	-52.3210992806125};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999999949391,	-3.34892972881350,	-2.90354997790204e-06,	9.40809286651173,	-0.00202102912071389,	-22.4079242663551,	-0.127494440141390,	40.3628453629183,	1.00129043567824,	-68.2979927953205,	62.5165406487472,	-123.679350496091,	370.733832235627,	-575.007281406772,	497.454284115259,	-251.516932661952,	70.4285943443364,	-8.51755028102381};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000000014117,	-9.78565713096596e-08,	-16.8554960167390,	-0.000503267485397529,	66.9861814233914,	-0.166317801874811,	-165.834210188812,	-9.44213154971351,	336.572920149574,	-118.732441877411,	-178.394196450287,	-143.063724754382,	-30.2757415524302,	1637.67048863007,	-3130.38431301248,	2845.93406002291,	-1451.77986202568,	403.984378290621,	-48.2190892668386};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.0000002)
    {
      // C = 18.579    Lambda_0 = 0.581538894879421;
      select_c = 18.579;
      Lambda_0 = 0.581538894879421;
      num_of_force_poly = 21;
      num_of_energy_poly = 19;
      num_of_Fourier_poly = 20;
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
      double poly_coeff_f[]={1.00000000001391,	-1.28633318441373e-08,	1.98077942914487e-06,	-20.4265557430431,	0.00394200706009684,	103.203332909006,	1.04450399343619,	-303.952248876184,	67.8704525227690,	221.869386826706,	1417.14471878573,	-5256.42757451275,	10756.5317620360,	-19536.3574847080,	30353.7589777164,	-35219.4823001557,	28675.6200323416,	-15892.9107975074,	5736.10077016188,	-1221.55644143221,	116.965522109065};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={1.00000000005217,	-3.43915091458899,	4.64213665401711e-06,	10.2129931628542,	0.00571173568370331,	-25.9092460458589,	0.915955162733418,	42.4006288697159,	34.7630277907244,	-207.725463967566,	400.092507505656,	-810.618700773072,	1514.32656925020,	-1991.54184036483,	1739.41639295321,	-998.867574034613,	365.080709288437,	-77.3722897236707,	7.25976546276111};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={1.99999999993243,	5.51109883076273e-08,	-17.8181566121596,	0.000403329572955524,	75.0664902481115,	0.200900168115801,	-201.890131676806,	19.1100848288943,	262.311238189096,	512.465159246175,	-2290.38301658670,	4608.28330492842,	-8744.58673117255,	14569.6694107304,	-17704.0598521805,	14744.4306847358,	-8232.67710285797,	2964.25639037565,	-625.365806147431,	58.9867306560664};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }
    else if(closet == 0.0000001)
    {
      // C = 19.299    Lambda_0 = 0.570587865098339;
      select_c = 19.299;
      Lambda_0 = 0.570587865098339;
      num_of_force_poly = 19;
      num_of_energy_poly = 18;
      num_of_Fourier_poly = 18;
      memory->create(force_poly_coeff, num_of_force_poly, "esp:force_poly_coeff");
      memory->create(energy_poly_coeff, num_of_energy_poly, "esp:force_poly_coeff");
      memory->create(Fourier_poly_coeff, num_of_Fourier_poly, "esp:force_poly_coeff");
      double poly_coeff_f[]={0.999999999446076,	4.23172355552837e-07,	-5.38070765192522e-05,	-21.6574981666958,	-0.0725929151074709,	115.376103424865,	-12.8311384183207,	-242.196183860689,	-540.052757427997,	2923.43021625385,	-6944.75751750092,	15407.1215589164,	-29663.9487694345,	41275.3086568348,	-39116.1649931860,	24738.2441470991,	-10053.0921969783,	2387.24812313145,	-252.955104163650};
      for(int i=0; i<num_of_force_poly; i++) force_poly_coeff[i] = poly_coeff_f[i];
      double poly_coeff_e[]={0.999999999704477,	-3.50515672756344,	-1.92878813886077e-05,	10.8308958216698,	-0.0168141249582918,	-28.3325153491859,	-1.79712035459171,	66.8399818759439,	-40.4413358286324,	24.6776690750191,	-207.208604930314,	328.177079765309,	-16.9557166242257,	-466.443658446301,	593.355537877069,	-356.900371522548,	111.263300988109,	-14.5431522070165};
      for(int i=0; i<num_of_energy_poly; i++) energy_poly_coeff[i] = poly_coeff_e[i];
      double poly_coeff_Fourier[]={2.00000000275849,	-1.82641509823001e-06,	-18.5383833200707,	-0.00877846730059684,	81.6480538709633,	-2.83339927159389,	-199.920726817654,	-169.907841596936,	1240.46490214984,	-2734.21323656804,	6356.49819835723,	-13481.0962707694,	19878.6975944560,	-19325.1827539063,	12301.1936402614,	-4972.96455121883,	1165.18443767022,	-121.020882876783};
      for(int i=0; i<num_of_Fourier_poly; i++) Fourier_poly_coeff[i] = poly_coeff_Fourier[i];
    }

    double spreading_options[15] = {0.1, 0.05, 0.01, 0.005,  0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001};
    int spreading_options_size = 15;
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
    
    if(spreading_closet == 0.00000001){
        spreading_select_c = 21.691;
        spreading_Lambda_0 = 0.538207997662044;

        Fourier_spreading_order = 20;

        if(order == 2){
          poly_order = 20;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[20][2] = {
            {  0.139045195142160,    0.139045195142160},
            {  1.67250092690857,    -1.67250092690857},
            {  7.84813628629667,     7.84813628629667},
            { 15.1668902489440,    -15.1668902489440},
            { -5.78253124664981,    -5.78253124664981},
            { -77.6385860673899,     77.6385860673899},
            { -96.7695999736188,    -96.7695999736188},
            { 111.246765174213,    -111.246765174213},
            { 355.653193077027,     355.653193077027},
            {  59.6393209077029,    -59.6393209077029},
            { -633.304019984274,   -633.304019984274},
            { -489.915888287335,    489.915888287335},
            { 689.870466649113,     689.870466649113},
            { 968.339391927714,    -968.339391927714},
            { -459.849879699889,   -459.849879699889},
            { -1172.75435723771,    1172.75435723771},
            { 145.393490719069,     145.393490719069},
            { 929.040731501340,    -929.040731501340},
            {  7.25282476217399,     7.25282476217399},
            { -383.519365673536,    383.519365673536}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 3){
          poly_order = 16;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[16][3] = {
            {  0.0113297340899756,   2.28211995691278,    0.0113297340899756},
            {  0.139968225172789,    2.00000000000000e-17, -0.139968225172789},
            {  0.740832820795678,   -10.6153102027218,    0.740832820795678},
            {  2.12720257428102,     8.00000000000000e-17, -2.12720257428102},
            {  3.29307775293292,     23.5452090759332,     3.29307775293292},
            {  1.59521105001699,     3.20000000000000e-16, -1.59521105001699},
            { -3.43867633512399,    -33.2091101435287,    -3.43867633512399},
            { -6.41600151856627,     1.28000000000000e-15,  6.41600151856627},
            { -2.01263064861508,     33.4988346480114,    -2.01263064861508},
            {  5.15499306894789,     5.12000000000000e-15, -5.15499306894789},
            {  5.33448516653254,    -25.6310102691392,     5.33448516653254},
            { -1.08310307176355,     2.04800000000000e-14,  1.08310307176355},
            { -4.16268873050883,     14.8242141280862,    -4.16268873050883},
            { -0.994781815504861,    8.19200000000000e-14,  0.994781815504861},
            {  1.51466503991982,    -5.24027483040044,     1.51466503991982},
            {  0.718650282612772,    0,                   -0.718650282612772}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 4){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[14][4] = {
            {  0.00199960692065666,   1.17440563657100,    1.17440563657100,    0.00199960692065666},
            {  0.0233708342211055,    3.17040158671781,   -3.17040158671781,   -0.0233708342211055},
            {  0.119211446814038,     0.904265358187821,   0.904265358187821,   0.119211446814038},
            {  0.343488514752065,    -4.82437569247556,    4.82437569247556,   -0.343488514752065},
            {  0.594475574372353,    -3.96155616785762,   -3.96155616785762,    0.594475574372353},
            {  0.571026207536070,     3.08240646508499,   -3.08240646508499,   -0.571026207536070},
            {  0.128884985691423,     4.30159622213492,    4.30159622213492,    0.128884985691423},
            { -0.367258713627947,    -0.878003173142821,   0.878003173142821,    0.367258713627947},
            { -0.416462745089333,    -2.70285591433988,   -2.70285591433988,   -0.416462745089333},
            { -0.0662557850623324,   -0.0761958367170705,  0.0761958367170705,   0.0662557850623324},
            {  0.190261510602948,     1.15618383393590,    1.15618383393590,    0.190261510602948},
            {  0.124253143337285,     0.136334950623677,  -0.136334950623677,  -0.124253143337285},
            { -0.0303513453782216,   -0.309803384956467,  -0.309803384956467,  -0.0303513453782216},
            { -0.0451871500699781,    0,                    0,                    0.0451871500699781}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 5){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[14][5] = {
            {  0.000565048233981659,  0.398506805745457,   2.28211995442809,   0.398506805745457,   0.000565048233981659},
            {  0.00618466696394540,   1.45217645478225,    2.00000000000000e-17, -1.45217645478225,  -0.00618466696394540},
            {  0.0296846667489110,    1.78607030707339,   -3.82151084359159,    1.78607030707339,    0.0296846667489110},
            {  0.0814915666976174,    0.239355686243862,   8.00000000000000e-17, -0.239355686243862, -0.0814915666976174},
            {  0.138588947705750,    -1.35178367531329,    3.05141690540067,   -1.35178367531329,    0.138588947705750},
            {  0.143735339867388,    -0.956682657827626,   3.20000000000000e-16,  0.956682657827626, -0.143735339867388},
            {  0.0729696277212052,    0.290491676631375,  -1.54864422476648,    0.290491676631375,   0.0729696277212052},
            { -0.0160784140364890,    0.543762180386669,   1.28000000000000e-15, -0.543762180386669,  0.0160784140364890},
            { -0.0491619866940106,    0.0575105371472513,   0.556556373830659,   0.0575105371472513, -0.0491619866940106},
            { -0.0245753699297146,   -0.165951694246727,   5.12000000000000e-15,  0.165951694246727,   0.0245753699297146},
            {  0.00521410566029568,  -0.0532953415747066,  -0.132462736723397,  -0.0532953415747066,  0.00521410566029568},
            {  0.0100765061165617,    0.0296240343556065,   0,                   -0.0296240343556065, -0.0100765061165617},
            {  0.00166298080193883,   0.0149534763986594,   0,                    0.0149534763986594,  0.00166298080193883},
            { -0.00162328813166823,   0,                    0,                    0,                    0.00162328813166823}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 6){
          poly_order = 13;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[13][6] = {
            {  0.000214773994302955,  0.139045194806291,   1.70301642109976,   1.70301642109976,   0.139045194806291,   0.000214773994302955},
            {  0.00220585632513333,   0.557500310186197,   2.00762181773799,  -2.00762181773799,  -0.557500310186197,  -0.00220585632513333},
            {  0.00993626435775327,   0.872015201990572,  -0.879850470653105, -0.879850470653105,  0.872015201990572,   0.00993626435775327},
            {  0.0257034197131045,    0.561736669496417,  -1.85294417278909,   1.85294417278909,  -0.561736669496417,  -0.0257034197131045},
            {  0.0417016365877258,   -0.0713914893540907,  0.0227758696214096,  0.0227758696214096, -0.0713914893540907,  0.0417016365877258},
            {  0.0427384865675052,   -0.319501357540568,   0.816559552785558,  -0.816559552785558,  0.319501357540568,  -0.0427384865675052},
            {  0.0247898053632039,   -0.132707915677974,   0.117006649931173,   0.117006649931173, -0.132707915677974,   0.0247898053632039},
            {  0.00288087331484802,   0.0508860315904125,  -0.227970607655928,   0.227970607655928, -0.0508860315904125, -0.00288087331484802},
            { -0.00698855357372397,   0.0539426166761875,  -0.0531447209768518,  -0.0531447209768518,  0.0539426166761875, -0.00698855357372397},
            { -0.00487157904361187,   0.00289356674190608,  0.0414050264079909,  -0.0414050264079909, -0.00289356674190608,  0.00487157904361187},
            { -0.000455810011213309, -0.00978196747938220,  0.0120950028365101,   0.0120950028365101, -0.00978196747938220, -0.000455810011213309},
            {  0.000889865404928969, -0.00230814334372553,  0,                    0,                    0.00230814334372553, -0.000889865404928969},
            {  0.000348512043046867,  0,                    0,                    0,                    0,                    0.000348512043046867}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 7){
          poly_order = 13;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[13][7] = {
            {  9.95502431136609e-05,  0.0540752242384053,  0.954211860166587,   2.28211995791377,   0.954211860166587,   0.0540752242384053,   9.95502431136609e-05},
            {  0.000963693733512654,  0.223704539340011,   1.69914617530266,    2.00000000000000e-17, -1.69914617530266,  -0.223704539340011,  -0.000963693733512654},
            {  0.00408398631501991,   0.380666700671947,   0.590124969870063,  -1.94975093521394,    0.590124969870063,   0.380666700671947,   0.00408398631501991},
            {  0.00994469893919076,   0.317874231686326,  -0.665584021543088,   8.00000000000000e-17,  0.665584021543088,  -0.317874231686326,  -0.00994469893919076},
            {  0.0152684772679940,    0.0915546383322558, -0.503985337272296,   0.794320502229708,  -0.503985337272296,   0.0915546383322558,  0.0152684772679940},
            {  0.0150597004846179,   -0.0591127610574819,  0.0730776753727631,  3.20000000000000e-16, -0.0730776753727631,  0.0591127610574819, -0.0150597004846179},
            {  0.00892140539381220,  -0.0578739538109523,  0.151852005705269,  -0.205770603456533,   0.151852005705269,  -0.0578739538109523,  0.00892140539381220},
            {  0.00204138131446515,  -0.00906198602977474, 0.0118663466937147,  1.28000000000000e-15, -0.0118663466937147, 0.00906198602977474, -0.00204138131446515},
            { -0.00116718546046122,   0.00924254171632238, -0.0271085378247386,  0.0380316915085583, -0.0271085378247386,  0.00924254171632238, -0.00116718546046122},
            { -0.00107845310638302,   0.00427178420955068, -0.00487669980561279, 5.12000000000000e-15, 0.00487669980561279,-0.00427178420955068, 0.00107845310638302},
            { -0.000226402737262155, -0.000490458605405308, 0.00311830047556860,-0.00496465472897434, 0.00311830047556860,-0.000490458605405308,-0.000226402737262155},
            {  0.000106375432082473, -0.000644263680250390, 0,                   0,                   0,                   0.000644263680250390, -0.000106375432082473},
            {  6.02237314240028e-05,  0,                    0,                   0,                   0,                   0,                    6.02237314240028e-05}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 8){
          poly_order = 13;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[13][8] = {
            { 5.30636915489310e-05, 0.0235425034280704, 0.496281588030282, 1.93663326919883, 1.93663326919883, 0.496281588030282, 0.0235425034280704, 5.30636915489310e-05 },
            { 0.000486432680430023, 0.0978058838813323, 1.04803062013515, 1.27649418606389, -1.27649418606389, -1.04803062013515, -0.0978058838813323, -0.000486432680430023 },
            { 0.00194655764418402, 0.171540167355239, 0.701956607523662, -0.875443381176531, -0.875443381176531, 0.701956607523662, 0.171540167355239, 0.00194655764418402 },
            { 0.00447067988815065, 0.157806946550597, -0.0327757658933265, -0.722004368122998, 0.722004368122998, 0.0327757658933265, -0.157806946550597, -0.00447067988815065 },
            { 0.00648697587424631, 0.0698824338382097, -0.248572018989343, 0.172202766022605, 0.172202766022605, -0.248572018989343, 0.0698824338382097, 0.00648697587424631 },
            { 0.00610134873259581, -0.00175679661207887, -0.0765505901262291, 0.195765159549736, -0.195765159549736, 0.0765505901262291, 0.00175679661207887, -0.00610134873259581 },
            { 0.00355536573722191, -0.0176413307376043, 0.0316611573605217, -0.0175681954210567, -0.0175681954210567, 0.0316611573605217, -0.0176413307376043, 0.00355536573722191 },
            { 0.000974009442165184, -0.00683201410782796, 0.0204089545409962, -0.0339344723318499, 0.0339344723318499, -0.0204089545409962, 0.00683201410782796, -0.000974009442165184 },
            { -0.000207712714434200, 0.000683678192296153, -0.00104159839053720, 0.000538244834871449, 0.000538244834871449, -0.00104159839053720, 0.000683678192296153, -0.000207712714434200 },
            { -0.000275477053493968, 0.00117897507530967, -0.00280374742259770, 0.00422491068334535, -0.00422491068334535, 0.00280374742259770, -0.00117897507530967, 0.000275477053493968 },
            { -7.43354815490038e-05, 0.000191119336129253, -0.000215159626845152, 0.000103140933456647, 0.000103140933456647, -0.000215159626845152, 0.000191119336129253, -7.43354815490038e-05 },
            { 1.30413751122660e-05, -9.00024816470335e-05, 0.000242123203520350, -0.000380832678331444, 0.000380832678331444, -0.000242123203520350, 9.00024816470335e-05, -1.30413751122660e-05 },
            { 1.16354250820905e-05, -3.23174060667952e-05, 0, 0, 0, 0, -3.23174060667952e-05, 1.16354250820905e-05 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[20] = {1.22825521303982,	-1.86585226973762e-08,	-12.8547999047692,	-4.09631326954707e-05,	64.1529205579755,	0.0331756804103367,	-204.325771582658,	8.93322810529360,	389.912548962283,	416.609076290921,	-2562.10786790420,	5581.35196346832,	-12301.9142285051,	24385.8580664781,	-34391.2156338978,	32610.0228039515,	-20524.0901285556,	8289.01503313593,	-1957.02179212551,	206.413191621512};
        for(int i=0; i<20; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        } 
    }
    else if(spreading_closet == 0.00000005){
        spreading_select_c = 20.018;
        spreading_Lambda_0 = 0.560247067159076;

        Fourier_spreading_order = 20;
        
        if(order == 2){
          poly_order = 20;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[20][2] = {
            {  0.170477246690630,   0.170477246690630 },
            {  1.88581045900646,  -1.88581045900646 },
            {  7.93971041829526,   7.93971041829526 },
            { 12.5117375518545,  -12.5117375518545 },
            { -12.2042535077721,  -12.2042535077721 },
            { -71.2024507726712,   71.2024507726712 },
            { -57.0541489801573,  -57.0541489801573 },
            { 128.898581679532,  -128.898581679532 },
            { 245.533220128861,   245.533220128861 },
            { -69.2173601910738,   69.2173601910738 },
            { -464.572949905499,  -464.572949905499 },
            { -145.996272777394,   145.996272777394 },
            { 556.383603203138,   556.383603203138 },
            { 396.529693792853,  -396.529693792853 },
            { -461.186164186069,  -461.186164186069 },
            { -511.169992123933,   511.169992123933 },
            { 260.169761192290,   260.169761192290 },
            { 409.993459006663,  -409.993459006663 },
            { -80.1693934936993,  -80.1693934936993 },
            { -169.156118798011,   169.156118798011 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 3){
          poly_order = 18;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[18][3] = {
            {  0.0170008022536889,   2.23589767624799,   0.0170008022536889 },
            {  0.193055017279687,   2.00000000000000e-17,  -0.193055017279687 },
            {  0.925718522643493,  -9.56864109173632,   0.925718522643493 },
            {  2.34527911333243,   8.00000000000000e-17,  -2.34527911333243 },
            {  2.97618500033489,  19.4470227742902,   2.97618500033489 },
            {  0.403001494723033,   3.20000000000000e-16,  -0.403001494723033 },
            { -4.22399645120432,  -25.0329354746558,  -4.22399645120432 },
            { -4.99402102989782,   1.28000000000000e-15,   4.99402102989782 },
            {  0.394103570844743,  22.9690308958417,   0.394103570844743 },
            {  5.03360707417276,   5.12000000000000e-15,  -5.03360707417276 },
            {  2.68271071523279,  -16.0267401759304,   2.68271071523279 },
            { -2.28721009056735,   2.04800000000000e-14,   2.28721009056735 },
            { -2.69752327227703,   8.82593179789239,  -2.69752327227703 },
            {  0.304148461393629,   8.19200000000000e-14,  -0.304148461393629 },
            {  1.42300021497926,  -3.80909877352548,   1.42300021497926 },
            {  0.241807039469467,   3.27680000000000e-13,  -0.241807039469467 },
            { -0.407499979767927,   1.05523468145187,  -0.407499979767927 },
            { -0.135299183895911,   0,   0.135299183895911 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 4){
          poly_order = 16;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[16][4] = {
            {  0.00345321955759124,   1.21343085982082,   1.21343085982082,   0.00345321955759124 },
            {  0.0370814283708822,   3.01353576526609,  -3.01353576526609,  -0.0370814283708822 },
            {  0.171607784858739,   0.534484108980837,   0.534484108980837,   0.171607784858739 },
            {  0.440120999087228,  -4.45480143984665,   4.45480143984665,  -0.440120999087228 },
            {  0.653539314663308,  -2.97647669870209,  -2.97647669870209,   0.653539314663308 },
            {  0.479458651924569,   2.88983738426599,  -2.88983738426599,  -0.479458651924569 },
            { -0.0618225230761429,   3.11845302791099,   3.11845302791099,  -0.0618225230761429 },
            { -0.441142714923949,  -0.996546821612534,   0.996546821612534,   0.441142714923949 },
            { -0.297131673606744,  -1.85627403906033,  -1.85627403906033,  -0.297131673606744 },
            {  0.0636529385595450,   0.127891169127325,  -0.127891169127325,  -0.0636529385595450 },
            {  0.179603226298498,   0.762205298356804,   0.762205298356804,   0.179603226298498 },
            {  0.0521643199976000,   0.0494929059250353,  -0.0494929059250353,  -0.0521643199976000 },
            { -0.0483965715431164,  -0.231895899947631,  -0.231895899947631,  -0.0483965715431164 },
            { -0.0323617615787266,  -0.0273450341774151,   0.0273450341774151,   0.0323617615787266 },
            {  0.00581914887590971,   0.0472531177365454,   0.0472531177365454,   0.00581914887590971 },
            {  0.00821994130110830,   0,   0,  -0.00821994130110830 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 5){
          poly_order = 15;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[15][5] = {
            {  0.00108157218882832,   0.448992858885830,   2.23589767623752,   0.448992858885830,   0.00108157218882832 },
            {  0.0108719492105527,   1.50494037319653,   2.00000000000000e-17,  -1.50494037319653,  -0.0108719492105527 },
            {  0.0473476433097267,   1.63149634049185,  -3.44471078729721,   1.63149634049185,   0.0473476433097267 },
            {  0.115974792063000,  -0.00346537401467027,   8.00000000000000e-17,   0.00346537401467027,  -0.115974792063000 },
            {  0.171152279094217,  -1.28819317244656,   2.52033372278109,  -1.28819317244656,   0.171152279094217 },
            {  0.144483555626087,  -0.682788676288815,   3.20000000000000e-16,   0.682788676288815,  -0.144483555626087 },
            {  0.0424478973001485,   0.353270160897128,  -1.16792544928715,   0.353270160897128,   0.0424478973001485 },
            { -0.0419506630685213,   0.400980594252124,   1.28000000000000e-15,  -0.400980594252124,   0.0419506630685213 },
            { -0.0476777906284566,  -0.0125010686753942,   0.385657189139033,  -0.0125010686753942,  -0.0476777906284566 },
            { -0.0105842424025402,  -0.124430940929189,   5.12000000000000e-15,   0.124430940929189,   0.0105842424025402 },
            {  0.0107329446361888,  -0.0208015957489778,  -0.0960921448485124,  -0.0208015957489778,   0.0107329446361888 },
            {  0.00711483443792815,   0.0254506721321377,   2.04800000000000e-14,  -0.0254506721321377,  -0.00711483443792815 },
            { -0.000353873213498446,   0.00673495213231945,   0.0167600622808095,   0.00673495213231945,  -0.000353873213498446 },
            { -0.00148247519609868,  -0.00340187296751537,   0,   0.00340187296751537,   0.00148247519609868 },
            { -0.000257563020568341,   0,   0,   0,  -0.000257563020568341 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 6){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[14][6] = {
            {  0.000444952960328055,   0.170477246725046,   1.70804669163437,   1.70804669163437,   0.170477246725046,   0.000444952960328055 },
            {  0.00419529862872804,   0.628603486880768,   1.85246526512088,  -1.85246526512088,  -0.628603486880768,  -0.00419529862872804 },
            {  0.0171409510945169,   0.882190043640706,  -0.899058899881684,  -0.899058899881684,   0.882190043640706,   0.0171409510945169 },
            {  0.0395867517058761,   0.463397682847838,  -1.59625636128169,   1.59625636128169,  -0.463397682847838,  -0.0395867517058761 },
            {  0.0559737850660795,  -0.150669768296584,   0.0937935789604923,   0.0937935789604923,  -0.150669768296584,   0.0559737850660795 },
            {  0.0476678673359244,  -0.293014592802821,   0.656152773675249,  -0.656152773675249,   0.293014592802821,  -0.0476678673359244 },
            {  0.0195107079757504,  -0.0782634218649707,   0.0600074358869159,   0.0600074358869159,  -0.0782634218649707,   0.0195107079757504 },
            { -0.00375216374220077,   0.0589460931633302,  -0.171564409867480,   0.171564409867480,  -0.0589460931633302,   0.00375216374220077 },
            { -0.00862141394415891,   0.0374191422505140,  -0.0297853601323951,  -0.0297853601323951,   0.0374191422505140,  -0.00862141394415891 },
            { -0.00325648008108715,  -0.00357199828651855,   0.0319953498472426,  -0.0319953498472426,   0.00357199828651855,   0.00325648008108715 },
            {  0.000697722303379722,  -0.00784088388222948,   0.00732826670970788,   0.00732826670970788,  -0.00784088388222948,   0.000697722303379722 },
            {  0.000897237157637676,  -0.000637781546515726,  -0.00418756208330251,   0.00418756208330251,   0.000637781546515726,  -0.000897237157637676 },
            {  0.000104339598919978,   0.000966018082408482,  -0.00112056076955014,  -0.00112056076955014,   0.000966018082408482,   0.000104339598919978 },
            { -0.000101998618447396,   0,   0,   0,   0,   0.000101998618447396 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 7){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[14][7] = {
            {  0.000219686979455657,   0.0715316262152358,   1.00244630387598,   2.23589767622016,   1.00244630387598,   0.0715316262152358,   0.000219686979455657 },
            {  0.00195162875871354,   0.272095131316054,   1.64209976753576,   2.00000000000000e-17,  -1.64209976753576,  -0.272095131316054,  -0.00195162875871354 },
            {  0.00749797425072101,   0.417805790897537,   0.453448457945783,  -1.75750549689999,   0.453448457945783,   0.417805790897537,   0.00749797425072101 },
            {  0.0162985605922243,   0.300252878456201,  -0.649400766101443,   8.00000000000000e-17,   0.649400766101443,  -0.300252878456201,  -0.0162985605922243 },
            {  0.0218477415012669,   0.0510851644982767,  -0.400961562440069,   0.656063169931033,  -0.400961562440069,   0.0510851644982767,   0.0218477415012669 },
            {  0.0180759036751711,  -0.0722997866148332,   0.0904484549622259,   3.20000000000000e-16,  -0.0904484549622259,   0.0722997866148332,  -0.0180759036751711 },
            {  0.00803370419432935,  -0.0471423491154899,   0.116685445622033,  -0.155105421414375,   0.116685445622033,  -0.0471423491154899,   0.00803370419432935 },
            {  0.000187829673694023,  -0.000895357200234540,   0.00109392337543897,   1.28000000000000e-15,  -0.00109392337543897,   0.000895357200234540,  -0.000187829673694023 },
            { -0.00187084246984815,   0.00879741066619457,  -0.0200681159343883,   0.0260738433610245,  -0.0200681159343883,   0.00879741066619457,  -0.00187084246984815 },
            { -0.000875970276899991,   0.00234535503450625,  -0.00248596282059813,   5.12000000000000e-15,   0.00248596282059813,  -0.00234535503450625,   0.000875970276899991 },
            {  9.26596455963491e-06,  -0.000755300223403310,   0.00224666646687677,  -0.00310595451687057,   0.00224666646687677,  -0.000755300223403310,   9.26596455963491e-06 },
            {  0.000136385178869602,  -0.000401525556867455,   0.000448364885187402,   0,  -0.000448364885187402,   0.000401525556867455,  -0.000136385178869602 },
            {  2.99638744536423e-05,   0,   0,   0,   0,   0,   2.99638744536423e-05 },
            { -8.53462825034512e-06,   0,   0,   0,   0,   0,   8.53462825034512e-06 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 8){
          poly_order = 13;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[13][8] = {
            {  0.000123336612448845,   0.0333009338293933,   0.549403533867175,   1.92248243484569,   1.92248243484569,   0.549403533867175,   0.0333009338293933,   0.000123336612448845 },
            {  0.00103720600899376,   0.127186694493143,   1.06720556244484,   1.16581204751290,  -1.16581204751290,  -1.06720556244484,  -0.127186694493143,  -0.00103720600899376 },
            {  0.00376056009792430,   0.201815314446279,   0.624639235602442,  -0.830215808152819,  -0.830215808152819,   0.624639235602442,   0.201815314446279,   0.00376056009792430 },
            {  0.00770656938265413,   0.162518002738921,  -0.0855095854429020,  -0.610030359042768,   0.610030359042768,   0.0855095854429020,  -0.162518002738921,  -0.00770656938265413 },
            {  0.00976944862426112,   0.0556283152632886,  -0.225443666716827,   0.160052362810163,   0.160052362810163,  -0.225443666716827,   0.0556283152632886,   0.00976944862426112 },
            {  0.00774624664223470,  -0.0116800129937839,  -0.0493846082428308,   0.152500768596894,  -0.152500768596894,   0.0493846082428308,   0.0116800129937839,  -0.00774624664223470 },
            {  0.00348482798617590,  -0.0173952991257155,   0.0312765141751399,  -0.0173677549487387,  -0.0173677549487387,   0.0312765141751399,  -0.0173952991257155,   0.00348482798617590 },
            {  0.000392814274341955,  -0.00419630643781500,   0.0140358309519544,  -0.0242903766988857,   0.0242903766988857,  -0.0140358309519544,   0.00419630643781500,  -0.000392814274341955 },
            { -0.000474807342545083,   0.00140501898786598,  -0.00198410442240697,   0.000994218941173794,   0.000994218941173794,  -0.00198410442240697,   0.00140501898786598,  -0.000474807342545083 },
            { -0.000256410431652648,   0.000890594641198303,  -0.00191613790220842,   0.00277031875293575,  -0.00277031875293575,   0.00191613790220842,  -0.000890594641198303,   0.000256410431652648 },
            { -1.99321934116758e-05,   1.45324717981676e-05,  -5.58568747699439e-06,   5.96027602828286e-08,   5.96027602828286e-08,  -5.58568747699439e-06,   1.45324717981676e-05,  -1.99321934116758e-05 },
            {  2.43754026385545e-05,  -8.06864135451946e-05,   0.000163919218024944,  -0.000229140850649401,   0.000229140850649401,  -0.000163919218024944,   8.06864135451946e-05,  -2.43754026385545e-05 },
            {  7.28661427729094e-06,   0,   0,   0,   0,   0,   0,   7.28661427729094e-06 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[20] = {1.25265511554422,	3.98474765163359e-08,	-12.0618126035244,	0.000313038026249116,	55.1473690203783,	0.167839782558153,	-161.775919647439,	17.2387935477670,	222.538609948285,	500.916232167234,	-2303.00000746607,	4900.48462520638,	-9746.98472601067,	16935.4576921721,	-21535.8551967861,	18849.7264888945,	-11098.7183938907,	4225.91846095604,	-945.222241764759,	94.7692183202598};
        for(int i=0; i<20; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        }  
    }
    else if(spreading_closet == 0.0000001){
        spreading_select_c = 19.299;
        spreading_Lambda_0 = 0.570587865098339;

        Fourier_spreading_order = 17;
        
        if(order == 2){
          poly_order = 17;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[17][2] = {
            {  0.185982721072679,   0.185982721072679},
            {  1.98007072682168,   -1.98007072682168},
            {  7.92630408794486,    7.92630408794486},
            { 11.2551692963573,   -11.2551692963573},
            { -14.5390944842152,   -14.5390944842152},
            { -67.1836322174507,    67.1836322174507},
            { -41.5651814831243,   -41.5651814831243},
            { 129.639712075593,   -129.639712075593},
            { 200.201268818733,    200.201268818733},
            { -103.192942314891,   103.192942314891},
            { -385.903601593271,  -385.903601593271},
            { -33.7817580209056,    33.7817580209056},
            { 461.988127169392,    461.988127169392},
            { 154.877355495166,   -154.877355495166},
            { -362.484660412103,  -362.484660412103},
            { -116.533225128066,   116.533225128066},
            { 151.189517235143,    151.189517235143}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 3){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[14][3] = {
            {  0.0202295810565314,   2.21511849578341,    0.0202295810565314},
            {  0.221038549915961,    2.00000000000000e-17, -0.221038549915961},
            {  1.01262838735440,   -9.12558528243285,     1.01262838735440},
            {  2.41804196903570,    8.00000000000000e-17, -2.41804196903570},
            {  2.77168281541858,   17.8180958799351,      2.77168281541858},
            { -0.115683518075224,   3.20000000000000e-16,  0.115683518075224},
            { -4.38487882371179,  -21.9837336992658,     -4.38487882371179},
            { -4.23449504404803,    1.28000000000000e-15,  4.23449504404803},
            {  1.24388370162731,   19.2012523721361,      1.24388370162731},
            {  4.64211344263823,    5.12000000000000e-15, -4.64211344263823},
            {  1.40285607484085,  -12.1954385321140,      1.40285607484085},
            { -2.39743774901640,    2.04800000000000e-14,  2.39743774901640},
            { -1.16821226097730,    4.67310585832168,    -1.16821226097730},
            {  0.600090837143539,   0,                   -0.600090837143539}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 4){
          poly_order = 13;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[13][4] = {
            {  0.00436489221689811,  1.22993244812056,    1.22993244812056,    0.00436489221689811},
            {  0.0450897798476373,   2.94028430987731,   -2.94028430987731,   -0.0450897798476373},
            {  0.199506767534737,    0.385213137459578,   0.385213137459578,   0.199506767534737},
            {  0.484512601470380,   -4.27764521108811,    4.27764521108811,   -0.484512601470380},
            {  0.667866122507796,   -2.59130407892581,   -2.59130407892581,    0.667866122507796},
            {  0.421491342540678,    2.77030733569100,   -2.77030733569100,   -0.421491342540678},
            { -0.142488592702170,    2.66951441212048,    2.66951441212048,   -0.142488592702170},
            { -0.448770585582730,   -0.997735593241549,   0.997735593241549,   0.448770585582730},
            { -0.232798354134756,   -1.51598686692706,   -1.51598686692706,   -0.232798354134756},
            {  0.108810781433744,    0.181296815640374,  -0.181296815640374,  -0.108810781433744},
            {  0.157956939417767,    0.493638213461296,   0.493638213461296,   0.157956939417767},
            {  0.00864479098680704,  0,                   0,                  -0.00864479098680704},
            { -0.0421329366108694,   0,                   0,                  -0.0421329366108694}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 5){
          poly_order = 12;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[12][5] = {
            {  0.00142894140763019,  0.472355509804349,   2.21511850982267,   0.472355509804349,   0.00142894140763019},
            {  0.0138149600483872,   1.52391961668576,    2.00000000000000e-17, -1.52391961668576,  -0.0138149600483872},
            {  0.0575254642400899,   1.55657864254804,   -3.28521246837383,    1.55657864254804,   0.0575254642400899},
            {  0.133596443977270,   -0.0996921089308762,  8.00000000000000e-17,  0.0996921089308762, -0.133596443977270},
            {  0.184210154215158,   -1.24511727548807,    2.30925568592138,   -1.24511727548807,   0.184210154215158},
            {  0.139856240274603,   -0.572492807416121,   3.20000000000000e-16,  0.572492807416121, -0.139856240274603},
            {  0.0262834282711583,   0.363708496974197,  -1.02576178150381,    0.363708496974197,   0.0262834282711583},
            { -0.0508223692544899,   0.339304592753496,   1.28000000000000e-15, -0.339304592753496,  0.0508223692544899},
            { -0.0435458903824737,  -0.0326588488498961,  0.321569186581399,   -0.0326588488498961, -0.0435458903824737},
            { -0.00420100944411644, -0.0921404438111442,  5.12000000000000e-15,  0.0921404438111442,  0.00420100944411644},
            {  0.0107983061865215,  -0.00815726071786571, -0.0675539545700445,  -0.00815726071786571, 0.0107983061865215},
            {  0.00437263646364272,  0,                    0,                    0,                   -0.00437263646364272}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 6){
          poly_order = 12;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[12][6] = {
            {  0.000608196118878935,  0.185982723384464,   1.70928588281718,   1.70928588281718,   0.185982723384464,   0.000608196118878935},
            {  0.00551433906976429,   0.660023760920627,   1.78452459575538,  -1.78452459575538,  -0.660023760920627,  -0.00551433906976429},
            {  0.0215383995582058,    0.880700308070192,  -0.902185393989165, -0.902185393989165,  0.880700308070192,   0.0215383995582058},
            {  0.0471789271024674,    0.416854229438515,  -1.48911188284797,   1.48911188284797,  -0.416854229438515,  -0.0471789271024674},
            {  0.0624812214058303,   -0.179494238382711,   0.116885013348833,   0.116885013348833, -0.179494238382711,   0.0624812214058303},
            {  0.0484953652622185,   -0.276453875081738,   0.592201278278370,  -0.592201278278370,  0.276453875081738,  -0.0484953652622185},
            {  0.0159907333692795,   -0.0570035365032000,   0.0406984314299116,   0.0406984314299116, -0.0570035365032000,  0.0159907333692795},
            { -0.00654365893604941,   0.0592395514855115,  -0.149147555438191,   0.149147555438191, -0.0592395514855115,  0.00654365893604941},
            { -0.00867712066868803,   0.0303590486586527,  -0.0191405109359397,  -0.0191405109359397,  0.0303590486586527, -0.00867712066868803},
            { -0.00231251631246815,  -0.00528671199491759,  0.0246393337855840,  -0.0246393337855840,  0.00528671199491759,  0.00231251631246815},
            {  0.00105033737156235,  -0.00592525749349848,  0,                    0,                   -0.00592525749349848,  0.00105033737156235},
            {  0.000696838457386662,  0,                    0,                    0,                    0,                   -0.000696838457386662}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 7){
          poly_order = 12;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[12][7] = {
            { 0.000308549514274869, 0.0806275163169964, 1.02336418406966, 2.21511851134952, 1.02336418406966, 0.0806275163169964, 0.000308549514274869 },
            { 0.00263537048955005, 0.295152159154453, 1.61364951528337, 2.00000000000000e-17, -1.61364951528337, -0.295152159154453, -0.00263537048955005 },
            { 0.00967669309865393, 0.432141248822542, 0.396246345504327, -1.67612903142675, 0.396246345504327, 0.432141248822542, 0.00967669309865393 },
            { 0.0199499352783030, 0.288770473652428, -0.637405393432832, 8.00000000000000e-17, 0.637405393432832, -0.288770473652428, -0.0199499352783030 },
            { 0.0250744442100265, 0.0333071181569188, -0.358935926292050, 0.601123298664349, -0.358935926292050, 0.0333071181569188, 0.0250744442100265 },
            { 0.0190203169364671, -0.0756500246276808, 0.0944023350310391, 3.20000000000000e-16, -0.0944023350310391, 0.0756500246276808, -0.0190203169364671 },
            { 0.00717701455265880, -0.0416799642251275, 0.102660001845982, -0.136273154278240, 0.102660001845982, -0.0416799642251275, 0.00717701455265880 },
            { -0.000697908710162143, 0.00191713448611350, -0.00210328656925558, 1.28000000000000e-15, 0.00210328656925558, -0.00191713448611350, 0.000697908710162143 },
            { -0.00203347779208835, 0.00819648563610179, -0.0172934632299832, 0.0219291899943066, -0.0172934632299832, 0.00819648563610179, -0.00203347779208835 },
            { -0.000711837991033693, 0.00162995357889283, -0.00159156999293074, 5.12000000000000e-15, 0.00159156999293074, -0.00162995357889283, 0.000711837991033693 },
            { 9.53759262428688e-05, -0.000772712461971636, 0.00190066214210649, -0.00250372439302193, 0.00190066214210649, -0.000772712461971636, 9.53759262428688e-05 },
            {0.000121892118853495, -0.000304861176843474, 0.000308125566419903, 0, -0.000308125566419903, 0.000304861176843474, -0.000121892118853495 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 8){
          poly_order = 12;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[12][8] = {
            { 0.000177132748107828, 0.0386323189600583, 0.573637001138312, 1.91539308909241, 1.91539308909241, 0.573637001138312, 0.0386323189600583, 0.000177132748107828 },
            { 0.00143193332830436, 0.141984291216317, 1.07254139973590, 1.11811175670876, -1.11811175670876, -1.07254139973590, -0.141984291216317, -0.00143193332830436 },
            { 0.00496046865491600, 0.215100556997406, 0.588827525678793, -0.808890668410764, -0.808890668410764, 0.588827525678793, 0.215100556997406, 0.00496046865491600 },
            { 0.00963950139087406, 0.162571644925943, -0.105316160312677, -0.564406615896857, 0.564406615896857, 0.105316160312677, -0.162571644925943, -0.00963950139087406 },
            { 0.0114627861322339, 0.0483379299740051, -0.213614609291155, 0.153829916828688, 0.153829916828688, -0.213614609291155, 0.0483379299740051, 0.0114627861322339 },
            { 0.00835944555353226, -0.0154272208588506, -0.0390070012744259, 0.135875026016488, -0.135875026016488, 0.0390070012744259, 0.0154272208588506, -0.00835944555353226 },
            { 0.00326484021714253, -0.0166944338533947, 0.0302712878921415, -0.0168605637036042, -0.0168605637036042, 0.0302712878921415, -0.0166944338533947, 0.00326484021714253 },
            { 8.49863062691932e-05, -0.00307203767065343, 0.0115898348399361, -0.0207757978795835, 0.0207757978795835, -0.0115898348399361, 0.00307203767065343, -8.49863062691932e-05 },
            { -0.000556654729845225, 0.00156393000743926, -0.00215716933071036, 0.00105605154937619, 0.00105605154937619, -0.00215716933071036, 0.00156393000743926, -0.000556654729845225 },
            { -0.000224140134927085, 0.000742390106011105, -0.00148658049846580, 0.00215337724084776, -0.00215337724084776, 0.00148658049846580, -0.000742390106011105, 0.000224140134927085 },
            { 2.92734883902690e-06, -3.02932929702182e-05, 4.58583714195715e-05, 0, 0, 4.58583714195715e-05, -3.02932929702182e-05, 2.92734883902690e-06 },
            { 2.49519122146324e-05, -7.08548005334819e-05, 0, 0, 0, 0, 7.08548005334819e-05, -2.49519122146324e-05 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[17] = {1.26391973518716,	3.99199243214347e-06,	-11.7160083712156,	0.0132257477756307,	51.2228258688252,	2.76508057288247,	-162.787278468025,	95.2775157838359,	-26.6857742433259,	656.233467248015,	-1202.97426502759,	22.3489427374285,	2163.75210363905,	-2933.82231301904,	1882.48443695863,	-623.642786951973,	86.2669038880223};
        for(int i=0; i<17; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        }  
    }
    else if(spreading_closet == 0.0000005){
       spreading_select_c = 17.617;
       spreading_Lambda_0 = 0.597205717649591;
        
       Fourier_spreading_order = 19;

       if(order == 2){
        poly_order = 19;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        double array[19][2] = {
            {  0.227659562430612,   0.227659562430612 },
            {  2.20252061417016,  -2.20252061417016 },
            {  7.74946191534664,   7.74946191534664 },
            {  8.11540670719874,  -8.11540670719874 },
            { -18.7555675584884,  -18.7555675584884 },
            { -55.4124881103715,   55.4124881103715 },
            { -10.7766168819238,  -10.7766168819238 },
            { 117.458903345960,  -117.458903345960 },
            { 107.477584805045,   107.477584805045 },
            { -130.100085574926,   130.100085574926 },
            { -222.111359797443,  -222.111359797443 },
            {  70.1414999255327,  -70.1414999255327 },
            { 275.713425635426,   275.713425635426 },
            {  12.7960247656166,  -12.7960247656166 },
            { -240.757248355400,  -240.757248355400 },
            { -52.8217902515805,   52.8217902515805 },
            { 149.004488670464,   149.004488670464 },
            {  32.4475502740307,  -32.4475502740307 },
            { -52.1749143718744,  -52.1749143718744 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 3){
        poly_order = 16;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[16][3] = {
            {  0.0303410830435758,   2.16407710980994,   0.0303410830435758 },
            {  0.301058013845917,   2.00000000000000e-17,  -0.301058013845917 },
            {  1.22865337097584,  -8.10591488720011,   1.22865337097584 },
            {  2.51402315598324,   8.00000000000000e-17,  -2.51402315598324 },
            {  2.13452951813364,  14.3148286479965,   2.13452951813364 },
            { -1.24645406264286,   3.20000000000000e-16,   1.24645406264286 },
            { -4.28308758602822,  -15.8974883118088,  -4.28308758602822 },
            { -2.32424586556047,   1.28000000000000e-15,   2.32424586556047 },
            {  2.39569613439052,  12.4952301538697,   2.39569613439052 },
            {  3.23947445592069,   5.12000000000000e-15,  -3.23947445592069 },
            { -0.120814924668548,  -7.39727435933548,  -0.120814924668548 },
            { -2.00151494587804,   2.04800000000000e-14,   2.00151494587804 },
            { -0.564321130931758,   3.33890792653256,  -0.564321130931758 },
            {  0.755583987637422,   8.19200000000000e-14,  -0.755583987637422 },
            {  0.296043937912239,  -0.963924081996083,   0.296043937912239 },
            { -0.163949974109375,   0,   0.163949974109375 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 4){
        poly_order = 15;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[15][4] = {
            {  0.00754039130516089,   1.26755903503214,   1.26755903503214,   0.00754039130516089 },
            {  0.0706925453689467,   2.75480346843146,  -2.75480346843146,  -0.0706925453689467 },
            {  0.279200670183814,   0.0622957606952461,   0.0622957606952461,   0.279200670183814 },
            {  0.589097562573973,  -3.82607116964954,   3.82607116964954,  -0.589097562573973 },
            {  0.663593923689343,  -1.78728645996174,  -1.78728645996174,   0.663593923689343 },
            {  0.246446888231230,   2.42492026389273,  -2.42492026389273,  -0.246446888231230 },
            { -0.304939848145722,   1.78467580466074,   1.78467580466074,  -0.304939848145722 },
            { -0.402183430521093,  -0.911288326761498,   0.911288326761498,   0.402183430521093 },
            { -0.0781584526452369,  -0.964177190883630,  -0.964177190883630,  -0.0781584526452369 },
            {  0.152169942290561,   0.215161100995129,  -0.215161100995129,  -0.152169942290561 },
            {  0.0954408397172166,   0.349417489077794,   0.349417489077794,   0.0954408397172166 },
            { -0.0214568095764065,  -0.0278917068576448,   0.0278917068576448,   0.0214568095764065 },
            { -0.0355683855223106,  -0.0807144862101268,  -0.0807144862101268,  -0.0355683855223106 },
            { -0.00146039201354142,   0,   0,   0.00146039201354142 },
            {  0.00700932007748634,   0,   0,   0.00700932007748634 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
       }
       else if(order == 5){
        poly_order = 14;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[14][5] = {
            {  0.00273767518755672,   0.531093860500127,   2.16407710914869,   0.531093860500127,   0.00273767518755672 },
            {  0.0240077939366008,   1.55735778407313,   2.00000000000000e-17,  -1.55735778407313,  -0.0240077939366008 },
            {  0.0892460308156200,   1.36249176152518,  -2.91812915320474,   1.36249176152518,   0.0892460308156200 },
            {  0.180762213286822,  -0.299886526915435,   8.00000000000000e-17,   0.299886526915435,  -0.180762213286822 },
            {  0.208071792669135,  -1.11160025195819,   1.85519168353363,  -1.11160025195819,   0.208071792669135 },
            {  0.115015562216134,  -0.341381216760920,   3.20000000000000e-16,   0.341381216760920,  -0.115015562216134 },
            { -0.0146831352533857,   0.354414254290534,  -0.741534965924081,   0.354414254290534,  -0.0146831352533857 },
            { -0.0622383146321594,   0.220116272739296,   1.28000000000000e-15,  -0.220116272739296,   0.0622383146321594 },
            { -0.0285645573428506,  -0.0558852408386070,   0.208461828701678,  -0.0558852408386070,  -0.0285645573428506 },
            {  0.00706822925634187,  -0.0669099427948779,   5.12000000000000e-15,   0.0669099427948779,  -0.00706822925634187 },
            {  0.0102945481549708,   0.00256106080446727,  -0.0395764572121635,   0.00256106080446727,   0.0102945481549708 },
            {  0.00144322545629905,   0.0120378831440746,   0,  -0.0120378831440746,  -0.00144322545629905 },
            { -0.00161329516095066,   0,   0,   0,  -0.00161329516095066 },
            { -0.000545361679542111,   0,   0,   0,   0.000545361679542111 }
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       else if(order == 6){
        poly_order = 13;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[13][6] = {
            {  0.00126175089100232,   0.227659562418938,   1.70967730891438,   1.70967730891438,   0.227659562418938,   0.00126175089100232 },
            {  0.0103713626991231,   0.734173544928401,   1.62280082418836,  -1.62280082418836,  -0.734173544928401,  -0.0103713626991231 },
            {  0.0361485368017090,   0.861051336381836,  -0.897201340205343,  -0.897201340205343,   0.861051336381836,   0.0361485368017090 },
            {  0.0691340392402060,   0.300570212389431,  -1.24734207832189,   1.24734207832189,  -0.300570212389431,  -0.0691340392402060 },
            {  0.0770546346068664,  -0.231550930372431,   0.154540448372233,   0.154540448372233,  -0.231550930372431,   0.0770546346068664 },
            {  0.0459103063297646,  -0.228026885636668,   0.455604675096581,  -0.455604675096581,   0.228026885636668,  -0.0459103063297646 },
            {  0.00529984080014553,  -0.0147698162915939,   0.00952381011856011,   0.00952381011856011,  -0.0147698162915939,   0.00529984080014553 },
            { -0.0117555314486491,   0.0536375071070865,  -0.105487677324964,   0.105487677324964,  -0.0536375071070865,   0.0117555314486491 },
            { -0.00716678071795308,   0.0162791233347021,  -0.00990786755941921,  -0.00990786755941921,   0.0162791233347021,  -0.00716678071795308 },
            { -0.000136858520641610,  -0.00633380952976523,   0.0173793891021602,  -0.0173793891021602,   0.00633380952976523,   0.000136858520641610 },
            {  0.00133911449544470,  -0.00339052661351741,   0.00216988140698110,   0.00216988140698110,  -0.00339052661351741,   0.00133911449544470 },
            {  0.000330183643534099,   0,  -0.00202319351154756,   0.00202319351154756,   0,  -0.000330183643534099 },
            { -0.000108287660623319,   0,   0,   0,   0,  -0.000108287660623319 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       else if(order == 7){
        poly_order = 12;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[12][7] = {
            {  0.000682097204717491,   0.106529563461462,   1.07244874680936,   2.16407710994096,   1.07244874680936,   0.106529563461462,   0.000682097204717491 },
            {  0.00527917840112914,   0.354291120149136,   1.53728117885170,   2.00000000000000e-17,  -1.53728117885170,  -0.354291120149136,  -0.00527917840112914 },
            {  0.0172859542502916,   0.459582029172770,   0.267541838952930,  -1.48884151935530,   0.267541838952930,   0.459582029172770,   0.0172859542502916 },
            {  0.0311167167350638,   0.252365137515273,  -0.598098318549753,   8.00000000000000e-17,   0.598098318549753,  -0.252365137515273,  -0.0311167167350638 },
            {  0.0330346178380777,  -0.00711944423735826,  -0.267293240590673,   0.482924259736119,  -0.267293240590673,  -0.00711944423735826,   0.0330346178380777 },
            {  0.0196746702610735,  -0.0775376174770667,   0.0964262773513778,   3.20000000000000e-16,  -0.0964262773513778,   0.0775376174770667,  -0.0196746702610735 },
            {  0.00403061234722391,  -0.0279653091653601,   0.0730363332544756,  -0.0985049562552895,   0.0730363332544756,  -0.0279653091653601,   0.00403061234722391 },
            { -0.00263163763830410,   0.00656075189973564,  -0.00670041816266227,   1.28000000000000e-15,   0.00670041816266227,  -0.00656075189973564,   0.00263163763830410 },
            { -0.00197527352812142,   0.00617650436358505,  -0.0115912602278417,   0.0141977545229240,  -0.0115912602278417,   0.00617650436358505,  -0.00197527352812142 },
            { -0.000246272364595835,   0.000255676220215554,  -0.000177100418305834,   5.12000000000000e-15,   0.000177100418305834,  -0.000255676220215554,   0.000246272364595835 },
            {  0.000210227134462587,  -0.000657509742554296,   0.00120204498903798,  -0.00145537430052567,   0.00120204498903798,  -0.000657509742554296,   0.000210227134462587 },
            {  7.54775567592958e-05,   0,   0,   0,   0,   0,  -7.54775567592958e-05 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 
       else if(order == 8){
        poly_order = 12;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[12][8] = {
            {  0.000412560513604877,   0.0546015296331088,   0.633660475800737,   1.89612404507496,   1.89612404507496,   0.633660475800737,   0.0546015296331088,   0.000412560513604877 },
            {  0.00302076224634106,   0.182274337380862,   1.07690342689145,   1.00634529883174,  -1.00634529883174,  -1.07690342689145,  -0.182274337380862,  -0.00302076224634106 },
            {  0.00932436064330002,   0.245530573968288,   0.499789765240875,  -0.754650779525812,  -0.754650779525812,   0.499789765240875,   0.245530573968288,   0.00932436064330002 },
            {  0.0158139238753076,   0.156987888953409,  -0.143866454861518,  -0.463819517818313,   0.463819517818313,   0.143866454861518,  -0.156987888953409,  -0.0158139238753076 },
            {  0.0159106492260587,   0.0292109270137117,  -0.182448573798206,   0.137387445622048,   0.137387445622048,  -0.182448573798206,   0.0292109270137117,   0.0159106492260587 },
            {  0.00921889574139411,  -0.0222804607494767,  -0.0182441981773072,   0.101469303398450,  -0.101469303398450,   0.0182441981773072,   0.0222804607494767,  -0.00921889574139411 },
            {  0.00223120396761178,  -0.0138289352999589,   0.0264064543039857,  -0.0149775316744646,  -0.0149775316744646,   0.0264064543039857,  -0.0138289352999589,   0.00223120396761178 },
            { -0.000661245956160417,  -0.000707053182868294,   0.00680421017370566,  -0.0140365419460240,   0.0140365419460240,  -0.00680421017370566,   0.000707053182868294,   0.000661245956160417 },
            { -0.000619339933456129,   0.00160828798529926,  -0.00209500208211856,   0.00103191129615602,   0.00103191129615602,  -0.00209500208211856,   0.00160828798529926,  -0.000619339933456129 },
            { -0.000109470831876828,   0.000368152579316052,  -0.000868920966748635,   0.00131719157692695,  -0.00131719157692695,   0.000868920966748635,  -0.000368152579316052,   0.000109470831876828 },
            {  4.14393987145378e-05,  -8.64605726269563e-05,   0,   0,   0,   0,  -8.64605726269563e-05,   4.14393987145378e-05 },
            {  1.84623897228065e-05,   0,   0,   0,   0,   0,   0,  -1.84623897228065e-05 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       double Fourier_array[19] = {1.29239922359155,	-6.32349193230514e-08,	-10.8920149818708,	-0.000325211324320586,	43.2864444296512,	-0.107474512360050,	-107.162002142119,	-6.10150223100402,	217.493292109074,	-76.7248626106733,	-115.278251038169,	-92.4477372526276,	-19.5641575660310,	1058.26202262089,	-2022.85312201707,	1839.04148310780,	-938.139583181804,	261.054548529364,	-31.1591567890018};
       for(int i=0; i<19; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.000001){
        spreading_select_c = 16.894;
        spreading_Lambda_0 = 0.609850928341748;
 
        Fourier_spreading_order = 15;
        
        if(order == 2){
          poly_order = 15;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[15][2] = {
            { 0.248157922475164,  0.248157922475164},
            { 2.29714813956378,  -2.29714813956378},
            { 7.60407880934644,   7.60407880934644},
            { 6.71113934585765,  -6.71113934585765},
            {-19.9655521121115,  -19.9655521121115},
            {-49.6058293195275,   49.6058293195275},
            {-0.252934571119851, -0.252934571119851},
            {107.840002852732,  -107.840002852732},
            { 74.2006307314466,   74.2006307314466},
            {-127.469110375954,   127.469110375954},
            {-156.723261212418,  -156.723261212418},
            { 88.8819666592203,  -88.8819666592203},
            {171.365180975649,   171.365180975649},
            {-30.5745961786316,   30.5745961786316},
            {-90.4740712784486,  -90.4740712784486}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 3){
          poly_order = 12;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[12][3] = {
            { 0.0360914989749282,  2.14097796314341,   0.0360914989749282},
            { 0.342536429847670,   2.00000000000000e-17, -0.342536429847670},
            { 1.32432546651099,   -7.67508976272946,    1.32432546651099},
            { 2.51450041150811,    8.00000000000000e-17, -2.51450041150811},
            { 1.79932406058841,   12.9354985558810,     1.79932406058841},
            {-1.66296425217437,    3.20000000000000e-16,  1.66296425217437},
            {-4.04684757805916,  -13.6278172765600,    -4.04684757805916},
            {-1.50496612587061,    1.28000000000000e-15,  1.50496612587061},
            { 2.61722221348352,    9.78707641003694,     2.61722221348352},
            { 2.40056149967880,    5.12000000000000e-15, -2.40056149967880},
            {-0.669801943663206,  -4.13271558471505,    -0.669801943663206},
            {-1.18286555231796,    0,                   1.18286555231796}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 4){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[11][4] = {
          {0.00953136643802728,  1.28317517950283,   1.28317517950283,   0.00953136643802728},
          {0.0854447993834379,   2.66887781398602,  -2.66887781398602,  -0.0854447993834379},
          {0.320023543965462,   -0.0639763426412495, -0.0639763426412495,  0.320023543965462},
          {0.631400161569591,   -3.61820316318165,    3.61820316318165,  -0.631400161569591},
          {0.641829974585447,   -1.48437616267809,   -1.48437616267809,   0.641829974585447},
          {0.157774824523463,    2.25479680046494,   -2.25479680046494,  -0.157774824523463},
          {-0.356279395795008,   1.46494229572838,    1.46494229572838,  -0.356279395795008},
          {-0.352995359110524,  -0.842405603118769,   0.842405603118769,  0.352995359110524},
          {-0.0131531886966323, -0.754385625777511,  -0.754385625777511, -0.0131531886966323},
          {0.132186491574195,    0.187143640797634,  -0.187143640797634, -0.132186491574195},
          {0.0474173110261980,   0.222813244308654,   0.222813244308654,  0.0474173110261980}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 5){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");

          double array[11][5] = {
          {0.00361799715343252,  0.558143158216225,   2.14097813077618,   0.558143158216225,   0.00361799715343252},
          {0.0303298906348130,   1.56616778069361,    2.00000000000000e-17, -1.56616778069361, -0.0303298906348130},
          {0.106928975199587,    1.27156103091659,   -2.76304411099186,    1.27156103091659,   0.106928975199587},
          {0.202942706775863,   -0.373393255618010,   8.00000000000000e-17,  0.373393255618010, -0.202942706775863},
          {0.213614439128090,   -1.04196259203023,    1.67645934574121,   -1.04196259203023,   0.213614439128090},
          {0.0980703846014108,  -0.254831502605868,   3.20000000000000e-16,  0.254831502605868, -0.0980703846014108},
          {-0.0318790423006054,  0.337790938354738,  -0.633984665406665,    0.337790938354738, -0.0318790423006054},
          {-0.0623520425061038,  0.173252878457965,   1.28000000000000e-15, -0.173252878457965,  0.0623520425061038},
          {-0.0201279705861494, -0.0556738494939440,  0.151232482933658,   -0.0556738494939440, -0.0201279705861494},
          {0.0102185609386847,  -0.0463701116383588,  0,                   0.0463701116383588, -0.0102185609386847},
          {0.00749713293921550,  0,                    0,                   0,                   0.00749713293921550}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 6){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");

          double array[11][6] = {
            { 0.00172552569881273,  0.248157949443807,   1.70863283347569,   1.70863283347569,   0.248157949443807,   0.00172552569881273},
            { 0.0135551356363280,   0.765716893892693,   1.55215102799150,  -1.55215102799150,  -0.765716893892693,  -0.0135551356363280},
            { 0.0447970169219008,   0.844896782806958,  -0.889722892706905, -0.889722892706905,  0.844896782806958,   0.0447970169219008},
            { 0.0803339061660521,   0.248515298216806,  -1.14761302610400,   1.14761302610400,  -0.248515298216806,  -0.0803339061660521},
            { 0.0822974420097634,  -0.246502009321089,   0.164758772274353,   0.164758772274353, -0.246502009321089,   0.0822974420097634},
            { 0.0424467429982397,  -0.203482198494762,   0.401170249034365,  -0.401170249034365,  0.203482198494762,  -0.0424467429982397},
            {-0.000112499725284365,-5.92305273932432e-05,-0.00274814676599767,-0.00274814676599767,-5.92305273932432e-05,-0.000112499725284365},
            {-0.0130835430047173,   0.0457720662470325,  -0.0817654549403532,  0.0817654549403532, -0.0457720662470325,  0.0130835430047173},
            {-0.00516162035065672,  0.00978913512014715,  0,                   0,                   0.00978913512014715, -0.00516162035065672},
            { 0.000761024504002894, 0,                    0,                   0,                   0,                  -0.000761024504002894}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 7){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");

          double array[11][7] = {
            {  0.000958644964635771,   0.119997974550777,    1.09349058526087,    2.14097818807562,    1.09349058526087,    0.119997974550777,    0.000958644964635771},
            {  0.00708914147854757,   0.381805178374743,    1.50004379774421,    2.00000000000000e-17, -1.50004379774421,   -0.381805178374743,   -0.00708914147854757},
            {  0.0220024084476332,   0.467884496354470,    0.214955078807571,   -1.40972204357988,    0.214955078807571,    0.467884496354470,    0.0220024084476332},
            {  0.0371409191133990,   0.232668229684449,   -0.576634256921378,    8.00000000000000e-17,  0.576634256921378,   -0.232668229684449,   -0.0371409191133990},
            {  0.0363183044722039,  -0.0232273201601922,   -0.231182924820319,    0.436482341781993,   -0.231182924820319,   -0.0232273201601922,    0.0363183044722039},
            {  0.0190375093119575,  -0.0757535282821762,    0.0945608644301162,    3.20000000000000e-16, -0.0945608644301162,    0.0757535282821762,   -0.0190375093119575},
            {  0.00226743195951245,  -0.0219824003961510,    0.0616991294250795,   -0.0846213110327920,   0.0616991294250795,   -0.0219824003961510,    0.00226743195951245},
            { -0.00327462468046189,   0.00761702281729407,   -0.00756189694489351,   1.28000000000000e-15,  0.00756189694489351,   -0.00761702281729407,    0.00327462468046189},
            { -0.00175915594987813,   0.00481744259751285,   -0.00892028484171796,    0.0109009761994563,  -0.00892028484171796,    0.00481744259751285,   -0.00175915594987813},
            { -1.47789452498385e-05,   0,                     0,                     0,                     0,                     0,                     1.47789452498385e-05},
            {  0.000216798048023409,   0,                     0,                     0,                     0,                     0,                     0.000216798048023409}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 8)
        {
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          
          double array[11][8] = {
            {  0.000592989111461173,   0.0633127024423366,    0.660888390923135,    1.88656089945326,    1.88656089945326,    0.660888390923135,    0.0633127024423366,    0.000592989111461173},
            {  0.00414754957633464,   0.202181473721544,    1.07481407965173,    0.958275958666431,   -0.958275958666431,   -1.07481407965173,   -0.202181473721544,   -0.00414754957633464},
            {  0.0121304046650706,   0.257758483447206,    0.459605046097110,   -0.729492335870010,   -0.729492335870010,    0.459605046097110,    0.257758483447206,    0.0121304046650706},
            {  0.0192885683367984,   0.151820856266666,   -0.156769824410804,   -0.423307720235215,    0.423307720235215,    0.156769824410804,   -0.151820856266666,   -0.0192885683367984},
            {  0.0178951155886278,   0.0204275764296244,   -0.167914889622014,    0.129588044705475,    0.129588044705475,   -0.167914889622014,    0.0204275764296244,    0.0178951155886278},
            {  0.00920295945341686,  -0.0242246654247592,   -0.0108633544065350,    0.0884378514190375,   -0.0884378514190375,    0.0108633544065350,    0.0242246654247592,   -0.00920295945341686},
            {  0.00157325228379449,  -0.0121633430473507,    0.0242733195498908,   -0.0134666600221579,   -0.0134666600221579,    0.0242733195498908,   -0.0121633430473507,    0.00157325228379449},
            { -0.000944193500220649,   0.000252195342763896,    0.00477210804232550,   -0.0110971053301640,    0.0110971053301640,   -0.00477210804232550,   -0.000252195342763896,    0.000944193500220649},
            { -0.000582355018579348,   0.00145013196296182,   -0.00196592522479605,    0,                     0,                    -0.00196592522479605,    0.00145013196296182,   -0.000582355018579348},
            { -4.44539006593937e-05,   0,                     0,                     0,                     0,                     0,                     0,                     4.44539006593937e-05},
            {  4.76513256267713e-05,   0,                     0,                     0,                     0,                     0,                     0,                     4.76513256267713e-05}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[15] = {1.30567751696758,	1.00903402465331e-05,	-10.5324247661211,	0.0281078739958545,	39.4585122322499,	5.09416735855604,	-129.716953505041,	161.346424774266,	-367.515263364990,	1221.53278115840,	-2191.60472758937,	2192.84209065812,	-1271.93852562032,	404.875820389869,	-55.1756963608585};
        for(int i=0; i<Fourier_spreading_order; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        } 
    }
    else if(spreading_closet == 0.000005){
       spreading_select_c = 15.2;
       spreading_Lambda_0 = 0.642936586623448;
        
       Fourier_spreading_order = 17;

       if(order == 2){
        poly_order = 17;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        double array[17][2] = {
            {  0.303120132809666,   0.303120132809666 },
            {  2.50912525731017,  -2.50912525731017 },
            {  7.08206709354476,   7.08206709354476 },
            {  3.41359605102638,  -3.41359605102638 },
            { -21.2317479665041,  -21.2317479665041 },
            { -35.1979756324082,   35.1979756324082 },
            {  17.0255989381037,   17.0255989381037 },
            {  79.3702493946763,  -79.3702493946763 },
            {  18.7857626181904,   18.7857626181904 },
            { -99.6643164959371,   99.6643164959371 },
            { -61.3190232088256,  -61.3190232088256 },
            {  82.5981138352360,  -82.5981138352360 },
            {  77.6892323413122,   77.6892323413122 },
            { -46.5709613246564,   46.5709613246564 },
            { -59.9933077710639,  -59.9933077710639 },
            {  14.9156214887789,  -14.9156214887789 },
            {  24.5099613250137,   24.5099613250137 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 3){
        poly_order = 14;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[14][3] = {
            {  0.0540982160229673,   2.08370702336224,   0.0540982160229673 },
            {  0.458707875561574,   2.00000000000000e-17,  -0.458707875561574 },
            {  1.54265632435349,  -6.68476222438770,   1.54265632435349 },
            {  2.39281447515129,   8.00000000000000e-17,  -2.39281447515129 },
            {  0.911480462759658,  10.0132912941483,   0.911480462759658 },
            { -2.36020139413662,   3.20000000000000e-16,   2.36020139413662 },
            { -3.10200367282876,  -9.34259923721362,  -3.10200367282876 },
            {  0.0535293575213223,   1.28000000000000e-15,  -0.0535293575213223 },
            {  2.41451025221609,   6.10153746325054,   2.41451025221609 },
            {  1.02947303673269,   5.12000000000000e-15,  -1.02947303673269 },
            { -0.973906552383413,  -2.91028935709511,  -0.973906552383413 },
            { -0.770464370183532,   2.04800000000000e-14,   0.770464370183532 },
            {  0.203318681091110,   0.877837307489244,   0.203318681091110 },
            {  0.262392811574476,   0,  -0.262392811574476 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 4){
        poly_order = 13;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[13][4] = {
            {  0.0164731046075819,   1.31793096481877,   1.31793096481877,   0.0164731046075819 },
            {  0.131824309391476,   2.45267669897995,  -2.45267669897995,  -0.131824309391476 },
            {  0.430730788359090,  -0.326409218275791,  -0.326409218275791,   0.430730788359090 },
            {  0.711355234649572,  -3.10684936323758,   3.10684936323758,  -0.711355234649572 },
            {  0.535538662144463,  -0.878642294091632,  -0.878642294091632,   0.535538662144463 },
            { -0.0575626146612274,   1.83046730114253,  -1.83046730114253,   0.0575626146612274 },
            { -0.409349015653199,   0.863780930518429,   0.863780930518429,  -0.409349015653199 },
            { -0.215826512049530,  -0.665891529647997,   0.665891529647997,   0.215826512049530 },
            {  0.0835527670198774,  -0.410960484815543,  -0.410960484815543,   0.0835527670198774 },
            {  0.114158716469389,   0.166305236183940,  -0.166305236183940,  -0.114158716469389 },
            {  0.00821578570516834,   0.111414209404046,   0.111414209404046,   0.00821578570516834 },
            { -0.0266789941732952,  -0.0273890839284832,   0.0273890839284832,   0.0266789941732952 },
            { -0.00688114963486441,   0,   0,  -0.00688114963486441 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
       }
       else if(order == 5){
        poly_order = 12;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[12][5] = {
            {  0.00694036880774795,   0.625816816421159,   2.08370702517325,   0.625816816421159,   0.00694036880774795 },
            {  0.0518966903088969,   1.57080135814761,   2.00000000000000e-17,  -1.57080135814761,  -0.0518966903088969 },
            {  0.159657838774835,   1.04361793600357,  -2.40651461758212,   1.04361793600357,   0.159657838774835 },
            {  0.255374413078479,  -0.510662007250828,   8.00000000000000e-17,   0.510662007250828,  -0.255374413078479 },
            {  0.209271641007502,  -0.857671927927000,   1.29772571977626,  -0.857671927927000,   0.209271641007502 },
            {  0.0456908200357419,  -0.0882768539496119,   3.20000000000000e-16,   0.0882768539496119,  -0.0456908200357419 },
            { -0.0640230323729386,   0.281095519940881,  -0.435884842096695,   0.281095519940881,  -0.0640230323729386 },
            { -0.0510052671120837,   0.0883208225687467,   1.28000000000000e-15,  -0.0883208225687467,   0.0510052671120837 },
            { -0.00181223959925437,  -0.0535239072797843,   0.102239588079568,  -0.0535239072797843,  -0.00181223959925437 },
            {  0.0123182494240868,  -0.0236790788642325,   5.12000000000000e-15,   0.0236790788642325,  -0.0123182494240868 },
            {  0.00317969939835905,   0.00619303272188176,  -0.0163552584680673,   0.00619303272188176,   0.00317969939835905 },
            { -0.00139654894310937,   0,   0,   0,   0.00139654894310937 }
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       else if(order == 6){
        poly_order = 11;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[11][6] = {
            {  0.00358648208241525,   0.303120131402403,   1.70283327624818,   1.70283327624818,   0.303120131402403,   0.00358648208241525 },
            {  0.0251128039135128,   0.836375084320116,   1.38421074045368,  -1.38421074045368,  -0.836375084320116,  -0.0251128039135128 },
            {  0.0723801524338182,   0.786896676896233,  -0.859357633182494,  -0.859357633182494,   0.786896676896233,   0.0723801524338182 },
            {  0.109633194076782,   0.126429238737860,  -0.925853989479317,   0.925853989479317,  -0.126429238737860,  -0.109633194076782 },
            {  0.0890173918364111,  -0.262131649932834,   0.173785544184675,   0.173785544184675,  -0.262131649932834,   0.0890173918364111 },
            {  0.0282859058579677,  -0.144840010609368,   0.292146396789444,  -0.292146396789444,   0.144840010609368,  -0.0282859058579677 },
            { -0.0117879531952446,   0.0234849197394268,  -0.0134539405614309,  -0.0134539405614309,   0.0234849197394268,  -0.0117879531952446 },
            { -0.0130715235524138,   0.0362165479504211,  -0.0578842076686151,   0.0578842076686151,  -0.0362165479504211,   0.0130715235524138 },
            { -0.00228932769125306,   0.00225228818481562,  -0.00102969449070874,  -0.00102969449070874,   0.00225228818481562,  -0.00228932769125306 },
            {  0.00163800849793142,  -0.00475351384533509,   0.00759014727812424,  -0.00759014727812424,   0.00475351384533509,  -0.00163800849793142 },
            {  0.000690937367514355,   0,   0,   0,   0,   0.000690937367514355 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       else if(order == 7){
        poly_order = 11;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[11][7] = {
            {  0.00212430620393316,   0.158300163906726,   1.14218049947654,   2.08370702430635,   1.14218049947654,   0.158300163906726,   0.00212430620393316 },
            {  0.0139927534846742,   0.450257453035606,   1.40186367395085,   2.00000000000000e-17,  -1.40186367395085,  -0.450257453035606,  -0.0139927534846742 },
            {  0.0378405339607497,   0.476126971778659,   0.0999631664275763,  -1.22781339221351,   0.0999631664275763,   0.476126971778659,   0.0378405339607497 },
            {  0.0539792816938104,   0.177940980161193,  -0.516711567225657,   8.00000000000000e-17,   0.516711567225657,  -0.177940980161193,  -0.0539792816938104 },
            {  0.0421969323075771,  -0.0554130143877699,  -0.155509077504028,   0.337802283787554,  -0.155509077504028,  -0.0554130143877699,   0.0421969323075771 },
            {  0.0147999973775534,  -0.0660057644863501,   0.0851061914609862,   3.20000000000000e-16,  -0.0851061914609862,   0.0660057644863501,  -0.0147999973775534 },
            { -0.00222926818597334,  -0.00937636249481049,   0.0393965423603376,  -0.0578159224338407,   0.0393965423603376,  -0.00937636249481049,  -0.00222926818597334 },
            { -0.00390522553316292,   0.00832833166693940,  -0.00777956288021313,   1.28000000000000e-15,   0.00777956288021313,  -0.00832833166693940,   0.00390522553316292 },
            { -0.000934570737277590,   0.00266942369872138,  -0.00526894432120490,   0.00658042192145124,  -0.00526894432120490,   0.00266942369872138,  -0.000934570737277590 },
            {  0.000283373311123960,  -0.000496477113881060,   0,   0,   0,   0.000496477113881060,  -0.000283373311123960 },
            {  0.000158254401416064,   0,   0,   0,   0,   0,   0.000158254401416064 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 
       else if(order == 8){
        poly_order = 11;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[11][8] = {
            {  0.00138502480284775,   0.0893891158217448,   0.727930704402286,   1.86067511277210,   1.86067511277210,   0.727930704402286,   0.0893891158217448,   0.00138502480284775 },
            {  0.00862329019945226,   0.255102408293382,   1.05901667301272,   0.845761602506542,  -0.845761602506542,  -1.05901667301272,  -0.255102408293382,  -0.00862329019945226 },
            {  0.0219518795668247,   0.282122583266099,   0.362350804607785,  -0.666360566390869,  -0.666360566390869,   0.362350804607785,   0.282122583266099,   0.0219518795668247 },
            {  0.0294879679815502,   0.132548389994016,  -0.177519512788416,  -0.335104949973660,   0.335104949973660,   0.177519512788416,  -0.132548389994016,  -0.0294879679815502 },
            {  0.0219663698296286,  -9.36647587377276e-05,  -0.132462617147665,   0.110515500150515,   0.110515500150515,  -0.132462617147665,  -9.36647587377276e-05,   0.0219663698296286 },
            {  0.00786232473883601,  -0.0258427025676593,   0.00228085555623824,   0.0623940394392169,  -0.0623940394392169,  -0.00228085555623824,   0.0258427025676593,  -0.00786232473883601 },
            { -0.000281999328808372,  -0.00776754756417428,   0.0186862822253378,  -0.0112767538077293,  -0.0112767538077293,   0.0186862822253378,  -0.00776754756417428,  -0.000281999328808372 },
            { -0.00133619155107392,   0.00145224036328594,   0.00201501968434476,  -0.00697199574847549,   0.00697199574847549,  -0.00201501968434476,  -0.00145224036328594,   0.00133619155107392 },
            { -0.000366135438931226,   0.00104432228646964,  -0.00151498170563773,   0.000768755495891765,   0.000768755495891765,  -0.00151498170563773,   0.00104432228646964,  -0.000366135438931226 },
            {  5.53799325602117e-05,   0,   0,   0,   0,   0,   0,  -5.53799325602117e-05 },
            {  4.12928287482038e-05,   0,   0,   0,   0,   0,   0,   4.12928287482038e-05 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       double Fourier_array[17] = {1.33969148304542,	-3.51753979005421e-07,	-9.67019176541188,	-0.00141234684410133,	32.6218319902954,	-0.380585767561295,	-65.2112070849741,	-18.9431408419211,	181.044053404932,	-249.763744725664,	464.937350422977,	-983.051926092768,	1315.74677130998,	-1054.87135377196,	506.102123302345,	-135.656605441580,	15.7583508343821};
       for(int i=0; i<17; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.00001){
        spreading_select_c = 14.471;
        spreading_Lambda_0 = 0.658932096384228;

        Fourier_spreading_order = 13;
        
        if(order == 2){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");

          double array[14][2] = {
            {  0.330058275848655,    0.330058275848655},
            {  2.59300734274074,   -2.59300734274074},
            {  6.77495627666949,    6.77495627666949},
            {  2.03933704406884,   -2.03933704406884},
            { -21.0896395652529,  -21.0896395652529},
            { -28.9995792138649,   28.9995792138649},
            {  21.4456925071599,    21.4456925071599},
            {  66.0336228265467,   -66.0336228265467},
            {  2.07970223271238,    2.07970223271238},
            { -82.2519879932368,    82.2519879932368},
            { -25.9643917678565,   -25.9643917678565},
            {  64.9865639267627,   -64.9865639267627},
            {  21.3841376824370,    21.3841376824370},
            { -27.8259004735400,    27.8259004735400}
          };

          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 3){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[11][3] = {
          {0.0643312715422107, 2.05753056054679, 0.0643312715422107},
          {0.517457795964784, 2.00000000000000e-17, -0.517457795964784},
          {1.62828173520205, -6.26710077396846, 1.62828173520205},
          {2.28055884230530, 8.00000000000000e-17, -2.28055884230530},
          {0.507493847814145, 8.88030085863462, 0.507493847814145},
          {-2.52076050493969, 3.20000000000000e-16, 2.52076050493969},
          {-2.57377963746076, -7.79562276020565, -2.57377963746076},
          {0.608639929146746, 1.28000000000000e-15, -0.608639929146746},
          {2.07746598658849, 4.66650727534594, 2.07746598658849},
          {0.249124047352474, 5.12000000000000e-15, -0.249124047352474},
          {-0.757090806635162, -1.68881775298291, -0.757090806635162}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 4){
          poly_order = 10;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");

          double array[10][4] = {
          {0.0208274733831955,  1.33188411274594,   1.33188411274594,   0.0208274733831955},
          {0.158041968955993,   2.35314019889406,  -2.35314019889406,  -0.158041968955993},
          {0.483925948644324,  -0.423644568766340, -0.423644568766340,  0.483925948644324},
          {0.732202275594278,  -2.87826887179309,   2.87826887179309,  -0.732202275594278},
          {0.465277741649882,  -0.661957343021911, -0.661957343021911,  0.465277741649882},
          {-0.145323178105331,  1.62897519538663,  -1.62897519538663,   0.145323178105331},
          {-0.399749769350901,  0.652128233088410,  0.652128233088410, -0.399749769350901},
          {-0.144310370009467, -0.506473246132380,  0.506473246132380,  0.144310370009467},
          {0.100756369904169,  -0.254912562098416, -0.254912562098416,  0.100756369904169},
          {0.0740670751708447,  0,                  0,                 -0.0740670751708447}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 5){
          poly_order = 10;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[10][5] = {
          {0.00917791801680253,  0.656778145436459,   2.05752839006094,   0.656778145436459,   0.00917791801680253},
          {0.0650482888476476,   1.56482743510283,    2.00000000000000e-17, -1.56482743510283,  -0.0650482888476476},
          {0.187565414133370,    0.940182305738912,  -2.25587499849539,    0.940182305738912,   0.187565414133370},
          {0.276038205330994,   -0.553231184343558,   8.00000000000000e-17,  0.553231184343558,  -0.276038205330994},
          {0.198066520827795,   -0.768504086484978,   1.14518075177493,    -0.768504086484978,   0.198066520827795},
          {0.0184635150171117,  -0.0312104722047224,  3.20000000000000e-16,  0.0312104722047224, -0.0184635150171117},
          {-0.0723479620255310,   0.227743085753803, -0.326432755207087,    0.227743085753803,  -0.0723479620255310},
          {-0.0360012821535376,   0.0513792261678636,  0,                  -0.0513792261678636,  0.0360012821535376},
          {0.00534328943200278,   0,                   0,                   0,                  0.00534328943200278}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 6){
          poly_order = 9;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[9][6] = {
          {  0.00490939770318428,   0.330058432535878,    1.69869187741120,    1.69869187741120,    0.330058432535878,    0.00490939770318428},
          {  0.0325712945855313,    0.864336066529646,    1.31099666672187,   -1.31099666672187,   -0.864336066529646,   -0.0325712945855313},
          {  0.0879620426335378,    0.752766593316139,   -0.840762707252061,  -0.840762707252061,   0.752766593316139,    0.0879620426335378},
          {  0.122737332145052,    0.0755060413113723,   -0.835827588946772,   0.835827588946772,  -0.0755060413113723,  -0.122737332145052},
          {  0.0883736709142075,   -0.260333517968697,    0.172691205530799,    0.172691205530799,  -0.260333517968697,    0.0883736709142075},
          {  0.0196863430776978,   -0.118935873614369,    0.250045648245535,   -0.250045648245535,   0.118935873614369,   -0.0196863430776978},
          { -0.0159090064284870,    0.0294324080584108,   -0.0163591742006778,  -0.0163591742006778,  0.0294324080584108,  -0.0159090064284870},
          { -0.0108093775399924,    0.0279438191892306,   -0.0437773501121533,   0.0437773501121533, -0.0279438191892306,   0.0108093775399924},
          { -0.000502715952440291,  0,                     0,                     0,                     0,                    -0.000502715952440291}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 7){
          poly_order = 9;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[9][7] = {
              {  0.00298918279643723,   0.178173430953499,    1.16266914545820,    2.05753047598323,    1.16266914545820,    0.178173430953499,    0.00298918279643723},
              {  0.0186495595974477,   0.480904472193357,    1.35472950967696,    2.00000000000000e-17, -1.35472950967696,   -0.480904472193357,   -0.0186495595974477},
              {  0.0472341861919932,   0.473794019164890,    0.0546323377224630,  -1.15108367882841,    0.0546323377224630,   0.473794019164890,    0.0472341861919932},
              {  0.0620916592785675,   0.151307953485573,   -0.487239707021830,    8.00000000000000e-17,  0.487239707021830,  -0.151307953485573,   -0.0620916592785675},
              {  0.0432961207368179,  -0.0662042634134114,  -0.126778398614065,    0.299208924834946,   -0.126778398614065,  -0.0662042634134114,    0.0432961207368179},
              {  0.0117239007564396,  -0.0597012655093132,   0.0793290125803664,    3.20000000000000e-16, -0.0793290125803664,   0.0597012655093132,   -0.0117239007564396},
              { -0.00403046004388277,  -0.00389606182239549,  0.0294797809953460,   -0.0457588204974441,   0.0294797809953460,  -0.00389606182239549,  -0.00403046004388277},
              { -0.00355428349991723,   0.00761030565190289, -0.00737472620934685,   0,                   0.00737472620934685,  -0.00761030565190289,   0.00355428349991723},
              { -0.000453681408485746,   0,                     0,                     0,                     0,                     0,                    -0.000453681408485746}
            };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 8){
          poly_order = 10;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[10][8] = {
              { 0.00199358167106259, 0.103594991501485, 0.758104789911040, 1.84784681024034, 1.84784681024034, 0.758104789911040, 0.103594991501485, 0.00199358167106259 },
              { 0.0117526267208000, 0.280502709214834, 1.04695824865777, 0.797457119010294, -0.797457119010294, -1.04695824865777, -0.280502709214834, -0.0117526267208000 },
              { 0.0280047686352990, 0.289829818914905, 0.319739654371398, -0.637435449625127, -0.637435449625127, 0.319739654371398, 0.289829818914905, 0.0280047686352990 },
              { 0.0346634335368865, 0.121121827677310, -0.182168019687644, -0.300117590788020, 0.300117590788020, 0.182168019687644, -0.121121827677310, -0.0346634335368865 },
              { 0.0231053317543149, -0.00834474981890105, -0.117064819780855, 0.101986640416864, 0.101986640416864, -0.117064819780855, -0.00834474981890105, 0.0231053317543149 },
              { 0.00665017360634066, -0.0252627031455872, 0.00625340326394305, 0.0529010818408253, -0.0529010818408253, -0.00625340326394305, 0.0252627031455872, -0.00665017360634066 },
              { -0.00109957648258485, -0.00583695521890383, 0.0161980813636269, -0.0100756634390073, -0.0100756634390073, 0.0161980813636269, -0.00583695521890383, -0.00109957648258485 },
              { -0.00134952963761369, 0.00170894173788683, 0.00118031660269479, -0.00559167984961274, 0.00559167984961274, -0.00118031660269479, -0.00170894173788683, 0.00134952963761369 },
              { -0.000219859683902233, 0.000836447956870270, -0.00129235642447025, 0.000669911467545033, 0.000669911467545033, -0.00129235642447025, 0.000836447956870270, -0.000219859683902233 },
              { 7.93555772975688e-05, 0, 0, 0, 0, 0, 0, -7.93555772975688e-05 }
            };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        
        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[13] = {1.35577360342720,	-0.000215519240876851,	-9.27937887363161,	-0.272813269762981,	32.7789315852292,	-21.5802194950179,	35.4070518996063,	-270.458338967371,	597.821436871576,	-636.365504979391,	368.970996818321,	-112.468405456826,	14.0906946724415};
        for(int i=0; i<13; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        }        
    }
    else if(spreading_closet == 0.00005){
       spreading_select_c = 12.762;
       spreading_Lambda_0 = 0.701666211927114;
        
       Fourier_spreading_order = 14;

       if(order == 2){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
          double array[14][2] = {
            {  0.401873420145306,   0.401873420145306 },
            {  2.76002362833187,  -2.76002362833187 },
            {  5.85494968101396,   5.85494968101396 },
            { -0.906753679540894,   0.906753679540894 },
            { -19.1802234313127,  -19.1802234313127 },
            { -15.6050127414922,   15.6050127414922 },
            {  24.7721745116366,   24.7721745116366 },
            {  37.0412052118625,  -37.0412052118625 },
            { -15.6424540463204,  -15.6424540463204 },
            { -44.7409409498186,   44.7409409498186 },
            {  2.74621451478752,   2.74621451478752 },
            {  34.0625715381733,  -34.0625715381733 },
            {  2.00242555584984,   2.00242555584984 },
            { -14.2068042134777,   14.2068042134777 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
       else if(order == 3){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[11][3] = {
            {  0.0963058546029360,   1.99188805635894,   0.0963058546029360 },
            {  0.676298738673936,   2.00000000000000e-17,  -0.676298738673936 },
            {  1.78534846219502,  -5.30974638741510,   1.78534846219502 },
            {  1.86261616005931,   8.00000000000000e-17,  -1.86261616005931 },
            { -0.395910028107205,   6.51886444121058,  -0.395910028107205 },
            { -2.48124293806196,   3.20000000000000e-16,   2.48124293806196 },
            { -1.28081962523880,  -4.91413821866918,  -1.28081962523880 },
            {  1.15563154990403,   1.28000000000000e-15,  -1.15563154990403 },
            {  1.19788632551087,   2.51922543414696,   1.19788632551087 },
            { -0.215666191645462,   5.12000000000000e-15,   0.215666191645462 },
            { -0.460304555982091,  -0.796275535829921,  -0.460304555982091 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
       else if(order == 4){
        poly_order = 11;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[11][4] = {
            {  0.0360003372949896,   1.36139369678880,   1.36139369678880,   0.0360003372949896 },
            {  0.238197249674569,   2.10452936002966,  -2.10452936002966,  -0.238197249674569 },
            {  0.614766702403688,  -0.610311373492577,  -0.610311373492577,   0.614766702403688 },
            {  0.730150497902685,  -2.33795778200882,   2.33795778200882,  -0.730150497902685 },
            {  0.251944825731047,  -0.261165944673890,  -0.261165944673890,   0.251944825731047 },
            { -0.302895053452424,   1.21195148829324,  -1.21195148829324,   0.302895053452424 },
            { -0.311443368089075,   0.309227423371274,   0.309227423371274,  -0.311443368089075 },
            { -0.00638120643997310,  -0.389170163004093,   0.389170163004093,   0.00638120643997310 },
            {  0.106977186112916,  -0.113651873213820,  -0.113651873213820,   0.106977186112916 },
            {  0.0251859533769583,   0.0789895494840334,  -0.0789895494840334,  -0.0251859533769583 },
            { -0.0184274053511047,   0,   0,  -0.0184274053511047 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
       }
       else if(order == 5){
        poly_order = 10;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[10][5] = {
            {  0.0176260372942613,   0.733472709971199,   1.99188807499147,   0.733472709971199,   0.0176260372942613 },
            {  0.108806151151712,   1.52840675867043,   2.00000000000000e-17,  -1.52840675867043,  -0.108806151151712 },
            {  0.264579648832729,   0.691264474064479,  -1.91150961366911,   0.691264474064479,   0.264579648832729 },
            {  0.309881709615730,  -0.609710473122966,   8.00000000000000e-17,   0.609710473122966,  -0.309881709615730 },
            {  0.146451116058011,  -0.566619004362366,   0.844827737124849,  -0.566619004362366,   0.146451116058011 },
            { -0.0403446527533383,   0.0575770008246330,   3.20000000000000e-16,  -0.0575770008246330,   0.0403446527533383 },
            { -0.0729907644462423,   0.174765755339633,  -0.228818973246303,   0.174765755339633,  -0.0729907644462423 },
            { -0.0167067988957729,   0.0159674785107844,   1.28000000000000e-15,  -0.0159674785107844,   0.0167067988957729 },
            {  0.0114930897288343,  -0.0298999417929026,   0.0396201969019146,  -0.0298999417929026,   0.0114930897288343 },
            {  0.00548586692773957,  -0.00552939668496175,   0,   0.00552939668496175,  -0.00548586692773957 }
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       else if(order == 6){
        poly_order = 10;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[10][6] = {
            {  0.0102240872629908,   0.401873446863321,   1.68433115542586,   1.68433115542586,   0.401873446863321,   0.0102240872629908 },
            {  0.0590153522350725,   0.920008042870964,   1.13751871480947,  -1.13751871480947,  -0.920008042870964,  -0.0590153522350725 },
            {  0.134301161130993,   0.650548775344626,  -0.784107169010955,  -0.784107169010955,   0.650548775344626,   0.134301161130993 },
            {  0.149927842779869,  -0.0335972491571269,  -0.639368451225537,   0.639368451225537,   0.0335972491571269,  -0.149927842779869 },
            {  0.0752451490861697,  -0.236783969965508,   0.159945047776515,   0.159945047776515,  -0.236783969965508,   0.0752451490861697 },
            { -0.00264915795666746,  -0.0639973213099961,   0.167221041572725,  -0.167221041572725,   0.0639973213099961,   0.00264915795666746 },
            { -0.0205077356024635,   0.0339534560860536,  -0.0177159839292172,  -0.0177159839292172,   0.0339534560860536,  -0.0205077356024635 },
            { -0.00648126639975027,   0.0157106898052798,  -0.0256059783117711,   0.0256059783117711,  -0.0157106898052798,   0.00648126639975027 },
            {  0.00168784891394321,  -0.00233482261850575,   0,   0,  -0.00233482261850575,   0.00168784891394321 },
            {  0.00116640184181615,   0,   0,   0,   0,  -0.00116640184181615 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       else if(order == 7){
        poly_order = 10;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[10][7] = {
            {  0.00664123747883949,   0.234464184672992,   1.20879934164499,   1.99188799978497,   1.20879934164499,   0.234464184672992,   0.00664123747883949 },
            {  0.0360124085955129,   0.552974836141849,   1.23226570470908,   2.00000000000000e-17,  -1.23226570470908,  -0.552974836141849,  -0.0360124085955129 },
            {  0.0767601540296214,   0.451196507254637,  -0.0397039405529109,  -0.975250079080599,  -0.0397039405529109,   0.451196507254637,   0.0767601540296214 },
            {  0.0808819259395420,   0.0846924896027228,  -0.411242901737250,   8.00000000000000e-17,   0.411242901737250,  -0.0846924896027228,  -0.0808819259395420 },
            {  0.0403250739123075,  -0.0813108935628461,  -0.0709822423586962,   0.219712176191620,  -0.0709822423586962,  -0.0813108935628461,   0.0403250739123075 },
            {  0.00250494791652644,  -0.0419814763248410,   0.0632545521424909,   3.20000000000000e-16,  -0.0632545521424909,   0.0419814763248410,  -0.00250494791652644 },
            { -0.00676579120059900,   0.00293024252675938,   0.0169579507411473,  -0.0290642408857290,   0.0169579507411473,   0.00293024252675938,  -0.00676579120059900 },
            { -0.00245639180427933,   0.00562387728376304,  -0.00575800959815690,   0,   0.00575800959815690,  -0.00562387728376304,   0.00245639180427933 },
            {  0.000272600040288016,   0.000477445895832704,  -0.00197121404019692,   0,  -0.00197121404019692,   0.000477445895832704,   0.000272600040288016 },
            {  0.000288107107896002,   0,   0,   0,   0,   0,  -0.000288107107896002 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 
       else if(order == 8){
        poly_order = 9;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[9][8] = {
            {  0.00467117225391113,   0.145994817131479,   0.831540970066187,   1.81306461442427,   1.81306461442427,   0.831540970066187,   0.145994817131479,   0.00467117225391113 },
            {  0.0239103021083033,   0.345266895847224,   1.00445114175189,   0.684727847244013,  -0.684727847244013,  -1.00445114175189,  -0.345266895847224,  -0.0239103021083033 },
            {  0.0478762903983843,   0.297981676073428,   0.220248872089692,  -0.565699905904580,  -0.565699905904580,   0.220248872089692,   0.297981676073428,   0.0478762903983843 },
            {  0.0475183428548723,   0.0876006438971755,  -0.182679129588460,  -0.225338794958014,   0.225338794958014,   0.182679129588460,  -0.0876006438971755,  -0.0475183428548723 },
            {  0.0229590284872800,  -0.0244243792584573,  -0.0820765627361698,   0.0817948645403412,   0.0817948645403412,  -0.0820765627361698,  -0.0244243792584573,   0.0229590284872800 },
            {  0.00247199841653335,  -0.0211406811380015,   0.0118215258595279,   0.0344056885916623,  -0.0344056885916623,  -0.0118215258595279,   0.0211406811380015,  -0.00247199841653335 },
            { -0.00254343744256095,  -0.00181145783990621,   0.0102504304751522,  -0.00710203197504728,  -0.00710203197504728,   0.0102504304751522,  -0.00181145783990621,  -0.00254343744256095 },
            { -0.000953416279172406,   0.00175015847094313,   0,  -0.00313973746991891,   0.00313973746991891,   0,  -0.00175015847094313,   0.000953416279172406 },
            {  3.80627464310898e-05,   0.000379856493007618,   0,   0,   0,   0,   0.000379856493007618,   3.80627464310898e-05 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       double Fourier_array[14] = {1.39764057599911,	-2.70143879898014e-06,	-8.38263729032031,	-0.00212593370213860,	23.1661145094270,	0.120057051989591,	-41.1093417525732,	11.1026253662201,	4.93193924982573,	99.9519100180898,	-199.808484995740,	162.053711013862,	-63.3897898719980,	9.96843449632046};
       for(int i=0; i<14; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.0001)
    {   
        spreading_select_c = 1.202400000000000e+01;
        spreading_Lambda_0 = 7.228787365921187e-01;
        
        Fourier_spreading_order = 12;
        
        if(order == 2){
          poly_order = 14;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[14][2] = {
            { 0.436925603369186, 0.436925603369186 },
            { 2.81421204437975, -2.81421204437975 },
            { 5.37323845777872, 5.37323845777872 },
            { -1.99881890340167, 1.99881890340167 },
            { -17.7494805543100, -17.7494805543100 },
            { -10.6461123622318, 10.6461123622318 },
            { 23.7941237884981, 23.7941237884981 },
            { 26.5169558271876, -26.5169558271876 },
            { -17.5737316184124, -17.5737316184124 },
            { -31.3455596130507, 31.3455596130507 },
            { 7.19496523118930, 7.19496523118930 },
            { 23.0720859616709, -23.0720859616709 },
            { -1.15668190037645, -1.15668190037645 },
            { -9.33296050739027, 9.33296050739027 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
        else if (order == 3){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[11][3] = {
            { 0.114482138709968, 1.96140336166991, 0.114482138709968 },
            { 0.753423878300845, 2.00000000000000e-17, -0.753423878300845 },
            { 1.82427704425321, -4.90631902057942, 1.82427704425321 },
            { 1.61751236153857, 8.00000000000000e-17, -1.61751236153857 },
            { -0.730312962122425, 5.62279625496583, -0.730312962122425 },
            { -2.30036092449397, 3.20000000000000e-16, 2.30036092449397 },
            { -0.767813161207901, -3.93780366314045, -0.767813161207901 },
            { 1.18038914010931, 1.28000000000000e-15, -1.18038914010931 },
            { 0.824329245968024, 1.87140578845648, 0.824329245968024 },
            { -0.287756692204448, 5.12000000000000e-15, 0.287756692204448 },
            { -0.321405293931125, -0.552923592965669, -0.321405293931125 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 4){
          poly_order = 11;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[11][4] = {
            { 0.0455377337805148, 1.37239600709157, 1.37239600709157, 0.0455377337805148 },
            { 0.282181633065391, 1.99056704626269, -1.99056704626269, -0.282181633065391 },
            { 0.669767701929085, -0.671723308398094, -0.671723308398094, 0.669767701929085 },
            { 0.701075646198157, -2.10470197435996, 2.10470197435996, -0.701075646198157 },
            { 0.146050317170654, -0.131752468764341, -0.131752468764341, 0.146050317170654 },
            { -0.340103579271099, 1.03653878084921, -1.03653878084921, 0.340103579271099 },
            { -0.250507784003064, 0.206424705872717, 0.206424705872717, -0.250507784003064 },
            { 0.0353971621219476, -0.315909013046559, 0.315909013046559, -0.0353971621219476 },
            { 0.0912095669841213, -0.0746440796229094, -0.0746440796229094, 0.0912095669841213 },
            { 0.00947573840874755, 0.0610899774754516, -0.0610899774754516, -0.00947573840874755 },
            { -0.0167728014940405, 0, 0, -0.0167728014940405 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if (order == 5){
          poly_order = 10;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[10][5] = {
              { 0.0233339205573305, 0.768225203537777, 1.96140337297240, 0.768225203537777, 0.0233339205573305 },
              { 0.134817024945303, 1.50166874358479, 2.00000000000000e-17, -1.50166874358479, -0.134817024945303 },
              { 0.301586939498526, 0.582850834921895, -1.76627531993143, 0.582850834921895, 0.301586939498526 },
              { 0.314230802262132, -0.614863497612424, 8.00000000000000e-17, 0.614863497612424, -0.314230802262132 },
              { 0.113808432793519, -0.479599570684563, 0.728700057011682, -0.479599570684563, 0.113808432793519 },
              { -0.0613994858113315, 0.0806889597224333, 3.20000000000000e-16, -0.0806889597224333, 0.0613994858113315 },
              { -0.0655161863056110, 0.143211242662065, -0.183394106303171, 0.143211242662065, -0.0655161863056110 },
              { -0.00663120051198874, 0.00282651566939360, 1.28000000000000e-15, -0.00282651566939360, 0.00663120051198874 },
              { 0.0114658601636976, -0.0236761786836995, 0.0295435156720823, -0.0236761786836995, 0.0114658601636976 },
              { 0.00333264744463006, 0, 0, 0, -0.00333264744463006 }
            };
            for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
            }
        }
        else if (order == 6){
          poly_order = 10;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[10][6] = {
              { 0.0140173884742002, 0.436925609023204, 1.67578910137486, 1.67578910137486, 0.436925609023204, 0.0140173884742002 },
              { 0.0756853026169014, 0.938070808550103, 1.06195728063377, -1.06195728063377, -0.938070808550103, -0.0756853026169014 },
              { 0.158388896246332, 0.597026209418350, -0.754018885797286, -0.754018885797286, 0.597026209418350, 0.158388896246332 },
              { 0.157854519983782, -0.0740401608927336, -0.561299165208735, 0.561299165208735, 0.0740401608927336, -0.157854519983782 },
              { 0.0640108978327278, -0.219126200986481, 0.150833362743522, 0.150833362743522, -0.219126200986481, 0.0640108978327278 },
              { -0.0120193608737977, -0.0436555380816834, 0.137407613774961, -0.137407613774961, 0.0436555380816834, 0.0120193608737977 },
              { -0.0199360158029335, 0.0326185442510810, -0.0169344940019014, -0.0169344940019014, 0.0326185442510810, -0.0199360158029335 },
              { -0.00385240178705037, 0.0112638303023511, -0.0196825246734326, 0.0196825246734326, -0.0112638303023511, 0.00385240178705037 },
              { 0.00204950829154430, -0.00259886509560908, 0, 0, -0.00259886509560908, 0.00204950829154430 },
              { 0.000812921581231219, 0, 0, 0, 0, -0.000812921581231219 }
            };
            for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
            }
        }
        else if (order == 7){
          poly_order = 9;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[9][7] = {
              { 0.00936356085314836, 0.263614032374908, 1.22754721207453, 1.96140331649313, 1.22754721207453, 0.263614032374908, 0.00936356085314836 },
              { 0.0474683747547741, 0.582931835162375, 1.17407785744804, 2.00000000000000e-17, -1.17407785744804, -0.582931835162375, -0.0474683747547741 },
              { 0.0929818494708278, 0.433078124698666, -0.0745952451723644, -0.901153441297151, -0.0745952451723644, 0.433078124698666, 0.0929818494708278 },
              { 0.0875942094158511, 0.0555415607626939, -0.376143765522278, 8.00000000000000e-17, 0.376143765522278, -0.0555415607626939, -0.0875942094158511 },
              { 0.0360732918586959, -0.0832334860515628, -0.0513236263435188, 0.189534546271861, -0.0513236263435188, -0.0832334860515628, 0.0360732918586959 },
              { -0.00183814075050075, -0.0337508748537442, 0.0557182178223906, 3.20000000000000e-16, -0.0557182178223906, 0.0337508748537442, 0.00183814075050075 },
              { -0.00703483862071266, 0.00495332767149522, 0.0116130252821214, -0.0233673929507308, 0.0116130252821214, 0.00495332767149522, -0.00703483862071266 },
              { -0.00154329353307353, 0.00455614031246625, -0.00491877575677593, 0, 0.00491877575677593, -0.00455614031246625, 0.00154329353307353 },
              { 0.000442562448273066, 0, 0, 0, 0, 0, 0.000442562448273066 }
            };
            for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
            }
        }
        else if (order == 8){
          poly_order = 9;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[9][8] = {
              { 0.00673916274509376, 0.169078712582033, 0.864198912409876, 1.79569685747976, 1.79569685747976, 0.864198912409876, 0.169078712582033, 0.00673916274509376 },
              { 0.0322322109037615, 0.374820303743804, 0.979285826945936, 0.636362525322216, -0.636362525322216, -0.979285826945936, -0.374820303743804, -0.0322322109037615 },
              { 0.0592643962022050, 0.295971616642733, 0.178340091101455, -0.533111555276462, -0.533111555276462, 0.178340091101455, 0.295971616642733, 0.0592643962022050 },
              { 0.0526140134759284, 0.0707708619048642, -0.178426662650776, -0.196257616914181, 0.196257616914181, 0.178426662650776, -0.0707708619048642, -0.0526140134759284 },
              { 0.0212449768909676, -0.0293519784520121, -0.0680204002973060, 0.0731998831423300, 0.0731998831423300, -0.0680204002973060, -0.0293519784520121, 0.0212449768909676 },
              { 0.000342207552079241, -0.0183963644188093, 0.0126604932893403, 0.0279416992032766, -0.0279416992032766, -0.0126604932893403, 0.0183963644188093, -0.000342207552079241 },
              { -0.00281030898516934, -0.000349021747785798, 0.00820144790405756, -0.00603501385659471, -0.00603501385659471, 0.00820144790405756, -0.000349021747785798, -0.00281030898516934 },
              { -0.000692238542580055, 0.00157628851189366, 0, -0.00237242994669541, 0.00237242994669541, 0, -0.00157628851189366, 0.000692238542580055 },
              { 0.000109763409994207, 0, 0, 0, 0, 0, 0, 0.000109763409994207 }
            };
            for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
            }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[12] = {1.41785709600891,	-9.54551171283846e-05,	-7.97495441189934,	-0.105867001294493,	21.7242448073335,	-7.34615931797523,	-2.57051679307573,	-79.8299129133637,	175.604955829871,	-155.878621576525,	66.1558807994186,	-11.1967089066119};
        for(int i=0; i<12; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        }  
    }
    else if(spreading_closet == 0.0005){
       spreading_select_c = 10.29;
       spreading_Lambda_0 = 0.781415895482355;
        
       Fourier_spreading_order = 12;

       if(order == 2){
        poly_order = 12;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        double array[12][2] = {
            {  0.529582549088048,   0.529582549088048 },
            {  2.87943863410091,  -2.87943863410091 },
            {  4.06528035964125,   4.06528035964125 },
            { -3.94792477934746,   3.94792477934746 },
            { -13.3265969301022,  -13.3265969301022 },
            { -1.76479511852130,   1.76479511852130 },
            {  17.7171959892698,   17.7171959892698 },
            {  8.47500234336293,  -8.47500234336293 },
            { -13.6420748347465,  -13.6420748347465 },
            { -9.28852719160994,   9.28852719160994 },
            {  5.71792209542899,   5.71792209542899 },
            {  4.62636362869923,  -4.62636362869923 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 3){
        poly_order = 10;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[10][3] = {
            {    0.171164365695719,   1.88352246558425,   0.171164365695719 },
            {  0.948850092693148,   2.00000000000000e-17,  -0.948850092693148 },
            {  1.81119861718316,  -3.98420584469297,   1.81119861718316 },
            {  0.922466976879949,   8.00000000000000e-17,  -0.922466976879949 },
            { -1.26743742876128,   3.79956410686465,  -1.26743742876128 },
            { -1.58237841100896,   3.20000000000000e-16,   1.58237841100896 },
            {  0.137434472683522,  -2.15774064458379,   0.137434472683522 },
            {  0.877994750231651,   1.28000000000000e-15,  -0.877994750231651 },
            {  0.120748511953344,   0.716501184235222,   0.120748511953344 },
            { -0.247840457446999,   0,   0.247840457446999 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 4){
        poly_order = 9;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[9][4] = {
            {  0.0787952257250252,   1.39266149960060,   1.39266149960060,   0.0787952257250252 },
            {  0.410447283457589,   1.70753875361766,  -1.70753875361766,  -0.410447283457589 },
            {  0.774756640617509,  -0.766485554610705,  -0.766485554610705,   0.774756640617509 },
            {  0.552842225058833,  -1.57038552046355,   1.57038552046355,  -0.552842225058833 },
            { -0.0962721708854775,   0.0778039864756156,   0.0778039864756156,  -0.0962721708854775 },
            { -0.334537996969677,   0.663281159387072,  -0.663281159387072,   0.334537996969677 },
            { -0.0936699810231360,   0.0416212175335003,   0.0416212175335003,  -0.0936699810231360 },
            {  0.0719651322982969,  -0.157265782563337,   0.157265782563337,  -0.0719651322982969 },
            {  0.0362822258358297,   0,   0,   0.0362822258358297 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
       }
       else if(order == 5){
        poly_order = 9;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[9][5] = {
            {  0.0449388961930707,   0.852899935359108,   1.88352245245207,   0.852899935359108,   0.0449388961930707 },
            {  0.217844582386834,   1.40839403137475,   2.00000000000000e-17,  -1.40839403137475,  -0.217844582386834 },
            {  0.388037870985820,   0.334602838103712,  -1.43428994307121,   0.334602838103712,   0.388037870985820 },
            {  0.286704240168599,  -0.581122400482946,   8.00000000000000e-17,   0.581122400482946,  -0.286704240168599 },
            {  0.0224972313223909,  -0.289867672176600,   0.491690478772386,  -0.289867672176600,   0.0224972313223909 },
            { -0.0867780444464021,   0.100355194679319,   3.20000000000000e-16,  -0.100355194679319,   0.0867780444464021 },
            { -0.0356588806907020,   0.0726105252624816,  -0.0951176500465181,   0.0726105252624816,  -0.0356588806907020 },
            {  0.00868208988215419,  -0.00846165758746551,   0,   0.00846165758746551,  -0.00868208988215419 },
            {  0.00711475347783599,   0,   0,   0,   0.00711475347783599 }
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 6){
        poly_order = 9;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[9][6] = {
            {  0.0293153935183165,   0.529582326563983,   1.64883596095470,   1.64883596095470,   0.529582326563983,   0.0293153935183165 },
            {  0.132560415587469,   0.959812309158132,   0.883457591325943,  -0.883457591325943,  -0.959812309158132,  -0.132560415587469 },
            {  0.220677008173373,   0.451698867204189,  -0.670233823194242,  -0.670233823194242,   0.451698867204189,   0.220677008173373 },
            {  0.158335660564562,  -0.146215003363401,  -0.395455323095778,   0.395455323095778,   0.146215003363401,  -0.158335660564562 },
            {  0.0264219207851836,  -0.164410093280526,   0.123263944475161,   0.123263944475161,  -0.164410093280526,   0.0264219207851836 },
            { -0.0272894943736520,  -0.00723995858290255,   0.0808330321554099,  -0.0808330321554099,   0.00723995858290255,   0.0272894943736520 },
            { -0.0131732805329375,   0.0233386064534818,  -0.0132061318720596,  -0.0132061318720596,   0.0233386064534818,  -0.0131732805329375 },
            {  0.00109177201114374,   0.00364341967540435,  -0.00963052904235524,   0.00963052904235524,  -0.00364341967540435,  -0.00109177201114374 },
            {  0.00164918175043126,   0,   0,   0,   0,   0.00164918175043126 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 7){
        poly_order = 7;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[7][7] = {
            {  0.0209157149789144,   0.345747100792810,   1.26735644065600,   1.88352283784242,   1.26735644065600,   0.345747100792810,   0.0209157149789144 },
            {  0.0886429861593104,   0.644934762921485,   1.02454574936474,   2.00000000000000e-17,  -1.02454574936474,  -0.644934762921485,  -0.0886429861593104 },
            {  0.137867963237696,   0.368671686250706,  -0.140382698587668,  -0.731804033053391,  -0.140382698587668,   0.368671686250706,   0.137867963237696 },
            {  0.0941373286849855,  -0.00827969799896400,  -0.290706654700153,   8.00000000000000e-17,   0.290706654700153,   0.00827969799896400,  -0.0941373286849855 },
            {  0.0189348887785292,  -0.0764079256231608,  -0.0165616571350765,   0.128195534347969,  -0.0165616571350765,  -0.0764079256231608,   0.0189348887785292 },
            { -0.00995292627731053,  -0.0146869704692110,   0.0367997525446520,   3.20000000000000e-16,  -0.0367997525446520,   0.0146869704692110,   0.00995292627731053 },
            { -0.00510639217591806,   0.00628457600429727,   0.00447703601449709,  -0.0130755451416826,   0.00447703601449709,   0.00628457600429727,  -0.00510639217591806 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 
      else if(order == 8){
        poly_order = 7;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[7][8] = {
            {  0.0158913979295034,   0.237746887939385,   0.942085226526322,   1.74805587387623,   1.74805587387623,   0.942085226526322,   0.237746887939385,   0.0158913979295034 },
            {  0.0634317594409783,   0.444263980235362,   0.901961061503760,   0.523813557889781,  -0.523813557889781,  -0.901961061503760,  -0.444263980235362,  -0.0634317594409783 },
            {  0.0923982784429670,   0.274308434098669,   0.0855985910675063,  -0.453064019862052,  -0.453064019862052,   0.0855985910675063,   0.274308434098669,   0.0923982784429670 },
            {  0.0596495313849674,   0.0286963685043321,  -0.158544641234933,  -0.135683577021298,   0.135683577021298,   0.158544641234933,  -0.0286963685043321,  -0.0596495313849674 },
            {  0.0127519824716323,  -0.0345303261697442,  -0.0388729837744415,   0.0537849775284766,   0.0537849775284766,  -0.0388729837744415,  -0.0345303261697442,   0.0127519824716323 },
            { -0.00412177339773692,  -0.0104204879404977,   0.0120611450916623,   0.0155139230635897,  -0.0155139230635897,  -0.0120611450916623,   0.0104204879404977,   0.00412177339773692 },
            { -0.00230320268839248,   0.00146564977350939,   0.00424026592158429,  -0.00381281048053831,  -0.00381281048053831,   0.00424026592158429,   0.00146564977350939,  -0.00230320268839248 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       double Fourier_array[12] = {1.47181497705627,	-7.97584588332796e-05,	-7.00120076833831,	-0.0754192621079211,	15.7929293196923,	-4.42600106767411,	-2.97211752479528,	-40.2409959849833,	81.5055273240189,	-64.7899918481550,	24.3512026420036,	-3.61511407772611};
       for(int i=0; i<12; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.001){
       spreading_select_c = 9.5392;
       spreading_Lambda_0 = 0.811584854067189;
        
       Fourier_spreading_order = 9;

       if(order == 2){
        poly_order = 9;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        rho_coeff[0][0] = 0.574307467864709; rho_coeff[1][0] = 2.87319911087849; rho_coeff[2][0] = 3.43664626314204; rho_coeff[3][0] = -4.49033598130652; rho_coeff[4][0] = -11.082356194141; rho_coeff[5][0] = 1.0781823376851; rho_coeff[6][0] = 13.7042413406122; rho_coeff[7][0] = 1.73889457319428; rho_coeff[8][0] = -7.93106821091405;
        rho_coeff[0][1] = 0.574307467864709; rho_coeff[1][1] = -2.87319911087849; rho_coeff[2][1] = 3.43664626314204; rho_coeff[3][1] = 4.49033598130652; rho_coeff[4][1] = -11.082356194141; rho_coeff[5][1] = -1.0781823376851; rho_coeff[6][1] = 13.7042413406122; rho_coeff[7][1] = -1.73889457319428; rho_coeff[8][1] = -7.93106821091405;  
       }
       else if(order == 3){
        poly_order = 8;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[8][3] = {
            {  0.203294303013937,   1.84652649597325,    0.203294303013937},
            {  1.03553747405562,    2.00000000000000e-17, -1.03553747405562},
            {  1.74746657848630,   -3.59520505464129,     1.74746657848630},
            {  0.591836442495712,   8.00000000000000e-17, -0.591836442495712},
            { -1.35874466478501,    3.09453847392948,    -1.35874466478501},
            { -1.18308785790887,    3.20000000000000e-16,  1.18308785790887},
            {  0.341529784390629,  -1.37032082609339,     0.341529784390629},
            {  0.561433437488923,   0,                   -0.561433437488923}
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 4){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        rho_coeff[0][-1] = 0.099693018874845; rho_coeff[1][-1] = 0.476781923984876; rho_coeff[2][-1] = 0.800880039422003; rho_coeff[3][-1] = 0.450560633317604; rho_coeff[4][-1] = -0.192702230561618; rho_coeff[5][-1] = -0.264856686543029;
        rho_coeff[0][0] = 1.39835440671411; rho_coeff[1][0] = 1.57840096107687; rho_coeff[2][0] = -0.784972435014229; rho_coeff[3][0] = -1.34359649381689; rho_coeff[4][0] = 0.132862727353174; rho_coeff[5][0] = 0.476301673064559;
        rho_coeff[0][1] = 1.39835440671411; rho_coeff[1][1] = -1.57840096107687; rho_coeff[2][1] = -0.784972435014229; rho_coeff[3][1] = 1.34359649381689; rho_coeff[4][1] = 0.132862727353174; rho_coeff[5][1] = -0.476301673064559;
        rho_coeff[0][2] = 0.099693018874845; rho_coeff[1][2] = -0.476781923984876; rho_coeff[2][2] = 0.800880039422003; rho_coeff[3][2] = -0.450560633317604; rho_coeff[4][2] = -0.192702230561618; rho_coeff[5][2] = 0.264856686543029;
       }
       else if(order == 5){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][5] = {
            {  0.0595598657675920,   0.890425967360455,    1.84650649800725,   0.890425967360455,    0.0595598657675920},
            {  0.264740815145233,   1.35142274193086,     2.00000000000000e-17, -1.35142274193086,  -0.264740815145233},
            {  0.420257657410069,   0.231633586862903,   -1.29243028222988,    0.231633586862903,   0.420257657410069},
            {  0.254482485358121,  -0.518069958296299,    8.00000000000000e-17,  0.518069958296299,  -0.254482485358121},
            { -0.0244028978652551, -0.199375750338259,    0.378673947007919,   -0.199375750338259,  -0.0244028978652551},
            { -0.0801967399441297,  0,                     0,                    0,                  0.0801967399441297}
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 6){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][6] = {
            {  0.0402705341555265,   0.574308183206901,    1.63356683157656,    1.63356683157656,    0.574308183206901,    0.0402705341555265},
            {  0.166775916493942,   0.957517264976828,    0.805977849765258,   -0.805977849765258,   -0.957517264976828,   -0.166775916493942},
            {  0.247226452488057,   0.381372114791673,   -0.628013820854955,   -0.628013820854955,    0.381372114791673,    0.247226452488057},
            {  0.147501761794350,  -0.164356078971058,   -0.331423384426196,    0.331423384426196,    0.164356078971058,   -0.147501761794350},
            {  0.00449811917382132, -0.130312327545096,    0.105305339187279,    0.105305339187279,   -0.130312327545096,    0.00449811917382132},
            { -0.0283465192387882,   0,                    0.0589437030118117,  -0.0589437030118117,   0,                    0.0283465192387882}
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 7){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][7] = {
            {  0.0295668024093345,   0.387985662080701,    1.28212508803122,    1.84653690451238,    1.28212508803122,    0.387985662080701,    0.0295668024093345},
            {  0.114636165228633,   0.665705837410951,    0.953637158659451,    2.00000000000000e-17, -0.953637158659451,   -0.665705837410951,   -0.114636165228633},
            {  0.158558525131127,   0.330997342014618,   -0.162455780005929,   -0.660357700806400,  -0.162455780005929,    0.330997342014618,    0.158558525131127},
            {  0.0906473575333341,  -0.0341754385309345,  -0.244237306301142,    8.00000000000000e-17,  0.244237306301142,    0.0341754385309345,  -0.0906473575333341},
            {  0.00800347953919924, -0.0668457802800258,    0,                    0.102018306473347,    0,                   -0.0668457802800258,    0.00800347953919924},
            { -0.0114674279300732,   0,                     0,                     0,                     0,                    0,                     0.0114674279300732}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 8){
        poly_order = 7;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[7][8] = {
            { 0.0230007799571616, 0.274962806420494, 0.975753669316238, 1.72388116689257, 1.72388116689257, 0.975753669316238, 0.274962806420494, 0.0230007799571616 },
            { 0.0838991326097754, 0.472305905344473, 0.859883656210132, 0.475696196631383, -0.475696196631383, -0.859883656210132, -0.472305905344473, -0.0838991326097754 },
            { 0.108469768651625, 0.256500118114656, 0.0491895126484451, -0.416930616097437, -0.416930616097437, 0.0491895126484451, 0.256500118114656, 0.108469768651625 },
            { 0.0588673541810641, 0.0109653295007298, -0.146072117098857, -0.112956644785801, 0.112956644785801, 0.146072117098857, -0.0109653295007298, -0.0588673541810641 },
            { 0.00751476335328056, -0.0337920073551342, -0.0283726303282873, 0.0447706918061301, 0.0447706918061301, -0.0283726303282873, -0.0337920073551342, 0.00751476335328056 },
            { -0.00517340259201607, -0.00728461189959164, 0.0109488264872737, 0.0117780804195784, -0.0117780804195784, -0.0109488264872737, 0.00728461189959164, 0.00517340259201607 },
            { -0.00175462047333324, 0.00173019369220991, 0.00294874229450147, 0, 0, 0.00294874229450147, 0.00173019369220991, -0.00175462047333324 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       Fourier_spreading_coeff[0] = 1.49863661960137; Fourier_spreading_coeff[1] = -0.00153580553856105; Fourier_spreading_coeff[2] = -6.53782778468899; Fourier_spreading_coeff[3] = -0.182990744426709; Fourier_spreading_coeff[4] = 12.9149486310785; Fourier_spreading_coeff[5] = 3.40652565837866; Fourier_spreading_coeff[6] = -28.9955774028528; Fourier_spreading_coeff[7] = 24.3356788477811; Fourier_spreading_coeff[8] = -6.43673483655126;
    }
    else if(spreading_closet == 0.005){
       spreading_select_c = 7.7625;
       spreading_Lambda_0 = 0.89968068782467;
        
       Fourier_spreading_order = 9;

       if(order == 2){
        poly_order = 9;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        double array[9][2] = {
            {  0.690743588320004,   0.690743588320004 },
            {  2.74309881925916,  -2.74309881925916 },
            {  1.89199579495238,   1.89199579495238 },
            { -4.78537785848098,   4.78537785848098 },
            { -5.97526935744817,  -5.97526935744817 },
            {  3.39044578103301,  -3.39044578103301 },
            {  6.41457518013304,   6.41457518013304 },
            { -1.09120915381698,   1.09120915381698 },
            { -3.23238775365954,  -3.23238775365954 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 3){
        poly_order = 8;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[8][3] = {
            {  0.303368485643295,   1.74869148378209,   0.303368485643295 },
            {  1.22158256486055,   2.00000000000000e-17,  -1.22158256486055 },
            {  1.42478108785836,  -2.71291221731635,   1.42478108785836 },
            { -0.136671724583299,   8.00000000000000e-17,   0.136671724583299 },
            { -1.19913840191153,   1.81976246772789,  -1.19913840191153 },
            { -0.368813995701004,   3.20000000000000e-16,   0.368813995701004 },
            {  0.388639566118861,  -0.634629384376342,   0.388639566118861 },
            {  0.181569393802150,   0,  -0.181569393802150 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
      }
       else if(order == 4){
        poly_order = 8;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[8][4] = {
            {  0.172921195957461,   1.40141429522199,   1.40141429522199,   0.172921195957461 },
            {  0.650850087734230,   1.25926528297793,  -1.25926528297793,  -0.650850087734230 },
            {  0.771699618437574,  -0.769013288027655,  -0.769013288027655,   0.771699618437574 },
            {  0.165839755888734,  -0.867099987466562,   0.867099987466562,  -0.165839755888734 },
            { -0.280287659672457,   0.173849594062903,   0.173849594062903,  -0.280287659672457 },
            { -0.146795580047288,   0.247628743302726,  -0.247628743302726,   0.146795580047288 },
            {  0.0375114258966044,  -0.0197043989813046,  -0.0197043989813046,   0.0375114258966044 },
            {  0.0331497718872154,   0,   0,  -0.0331497718872154 }
        };
        for(int i=0; i<poly_order; i++){
            for(int j=0; j<order; j++){
              rho_coeff[i][j+(1-order)/2] = array[i][j];
            }
        }
       }
       else if(order == 5){
        poly_order = 7;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[7][5] = {
            {  0.115360842539684,   0.978632908586110,   1.74869657865801,   0.978632908586110,   0.115360842539684 },
            {  0.402036481749406,   1.18244970546236,   2.00000000000000e-17,  -1.18244970546236,  -0.402036481749406 },
            {  0.452252158337418,   0.0224991267114879,  -0.976877102194709,   0.0224991267114879,   0.452252158337418 },
            {  0.134096595845855,  -0.429535409190116,   8.00000000000000e-17,   0.429535409190116,  -0.134096595845855 },
            { -0.0828844753006562,  -0.0860560098442270,   0.237353216630709,  -0.0860560098442270,  -0.0828844753006562 },
            { -0.0501685678952450,   0.0676868441586336,   3.20000000000000e-16,  -0.0676868441586336,   0.0501685678952450 },
            {  0.00367222120047373,   0.0180861557013073,  -0.0321877100759330,   0.0180861557013073,   0.00367222120047373 }
      };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 6){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][6] = {
            {  0.0849177703274343,   0.690745078139529,   1.58608231887933,   1.58608231887933,   0.690745078139529,   0.0849177703274343 },
            {  0.274694821457781,   0.914370972353646,   0.623174026163042,  -0.623174026163042,  -0.914370972353646,  -0.274694821457781 },
            {  0.288538594388558,   0.209974002065839,  -0.516574687578253,  -0.516574687578253,   0.209974002065839,   0.288538594388558 },
            {  0.0915479532482982,  -0.177239887980833,  -0.202041906213558,   0.202041906213558,   0.177239887980833,  -0.0915479532482982 },
            { -0.0292096031167816,  -0.0706576363647395,   0.0729052178101172,   0.0729052178101172,  -0.0706576363647395,  -0.0292096031167816 },
            { -0.0208630428894432,   0.0137725117828559,   0.0279381681103645,  -0.0279381681103645,  -0.0137725117828559,   0.0208630428894432 }
        };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 7){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][7] = {
            {  0.0667347955879768,   0.506044401606457,   1.30785858458556,   1.74869452947018,   1.30785858458556,   0.506044401606457,   0.0667347955879768 },
            {  0.201356902754059,   0.688149422359748,   0.774669774531480,   2.00000000000000e-17,  -0.774669774531480,  -0.688149422359748,  -0.201356902754059 },
            {  0.196972066073624,   0.222318927560817,  -0.188844157057046,  -0.498258173045639,  -0.188844157057046,   0.222318927560817,   0.196972066073624 },
            {  0.0622305910834454,  -0.0684551313321198,  -0.168915889027575,   8.00000000000000e-17,   0.168915889027575,   0.0684551313321198,  -0.0622305910834454 },
            { -0.0117365823622578,  -0.0432854685925881,   0.00827994398404637,   0.0601905656901167,   0.00827994398404637,  -0.0432854685925881,  -0.0117365823622578 },
            { -0.00962772237940132,   0,   0.0161399208117827,   0,  -0.0161399208117827,   0,   0.00962772237940132 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 
      else if(order == 8){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][8] = {
            {  0.0549041082135395,   0.385229896430659,   1.05251884447941,   1.65553135255854,   1.65553135255854,   1.05251884447941,   0.385229896430659,   0.0549041082135395 },
            {  0.155292346411473,   0.523842906142800,   0.737598290332556,   0.363842366153843,  -0.363842366153843,  -0.737598290332556,  -0.523842906142800,  -0.155292346411473 },
            {  0.141678506323010,   0.192675150183174,  -0.0234484679104151,  -0.329017022146173,  -0.329017022146173,  -0.0234484679104151,   0.192675150183174,   0.141678506323010 },
            {  0.0433794568152615,  -0.0237470038820813,  -0.109508558764014,  -0.0675936106513315,   0.0675936106513315,   0.109508558764014,   0.0237470038820813,  -0.0433794568152615 },
            { -0.00510756206260237,  -0.0250946031745672,  -0.00927399484968394,   0.0284660785313300,   0.0284660785313300,  -0.00927399484968394,  -0.0250946031745672,  -0.00510756206260237 },
            { -0.00485933047524132,  -0.00138486947969944,   0.00723956324606188,   0.00541511411810595,  -0.00541511411810595,  -0.00723956324606188,   0.00138486947969944,   0.00485933047524132 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      } 

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       double Fourier_array[9] = {1.57326528613285,	0.000631163184483512,	-5.51281970393949,	0.240271210614578,	6.87334952522541,	5.14742500073996,	-18.0218377597664,	12.6142071521804,	-2.90811683752856};
       for(int i=0; i<9; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
       } 
    }
    else if(spreading_closet == 0.01){
       spreading_select_c = 6.9862;
       spreading_Lambda_0 = 0.948344618546107;
        
       Fourier_spreading_order = 6;
       
       if(order == 2){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        rho_coeff[0][0] = 0.747188653323122; rho_coeff[1][0] = 2.62218036427046; rho_coeff[2][0] = 1.11858976349623; rho_coeff[3][0] = -4.44568905233768; rho_coeff[4][0] = -2.73844958387071; rho_coeff[5][0] = 2.80779505939869;
        rho_coeff[0][1] = 0.747188653323122; rho_coeff[1][1] = -2.62218036427046; rho_coeff[2][1] = 1.11858976349623; rho_coeff[3][1] = 4.44568905233768; rho_coeff[4][1] = -2.73844958387071; rho_coeff[5][1] = -2.80779505939869;
       }
       else if(order == 3){
        poly_order = 5;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        rho_coeff[0][-1] = 0.360107573963262; rho_coeff[1][-1] = 1.28272473895721; rho_coeff[2][-1] = 1.19411756036346; rho_coeff[3][-1] = -0.411098945570772; rho_coeff[4][-1] = -0.877202401550827;
        rho_coeff[0][0] = 1.69987474072505; rho_coeff[1][0] =            2e-17; rho_coeff[2][0] = -2.32806224996523; rho_coeff[3][0] = 8e-17; rho_coeff[4][0] = 1.21805138914482;
        rho_coeff[0][1] = 0.360107573963262; rho_coeff[1][1] = -1.28272473895721; rho_coeff[2][1] = 1.19411756036346; rho_coeff[3][1] = 0.411098945570772; rho_coeff[4][1] = -0.877202401550827;
       }
       else if(order == 4){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        rho_coeff[0][-1] = 0.219190865006589; rho_coeff[1][-1] = 0.727057173670847; rho_coeff[2][-1] = 0.706134843543027; rho_coeff[3][-1] = 0.0361723481635011; rho_coeff[4][-1] = -0.253588085950057; rho_coeff[5][-1] = -0.0735575984980672;
        rho_coeff[0][0] = 1.39664617169758; rho_coeff[1][0] =  1.1142133286212; rho_coeff[2][0] = -0.734784497823199; rho_coeff[3][0] = -0.682155222886481; rho_coeff[4][0] = 0.158818816354296; rho_coeff[5][0] = 0.172258729502554;
        rho_coeff[0][1] = 1.39664617169758; rho_coeff[1][1] = -1.1142133286212; rho_coeff[2][1] = -0.734784497823199; rho_coeff[3][1] = 0.682155222886481; rho_coeff[4][1] = 0.158818816354296; rho_coeff[5][1] = -0.172258729502554;
        rho_coeff[0][2] = 0.219190865006589; rho_coeff[1][2] = -0.727057173670847; rho_coeff[2][2] = 0.706134843543027; rho_coeff[3][2] = -0.0361723481635011; rho_coeff[4][2] = -0.253588085950057; rho_coeff[5][2] = 0.0735575984980672;
       }
       else if(order == 5){
        poly_order = 5;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[5][5] = {
            {  0.153484355062347,   1.01586725977441,    1.69873484188879,   1.01586725977441,    0.153484355062347},
            {  0.470755232352950,   1.08744539641681,    2.00000000000000e-17, -1.08744539641681, -0.470755232352950},
            {  0.435683562857948,  -0.0628029920323665, -0.800001776466606,  -0.0628029920323665,  0.435683562857948},
            {  0.0593927461272228, -0.348843425977332,   0,                   0.348843425977332,  -0.0593927461272228},
            { -0.0884363030749882,  0,                    0,                   0,                 -0.0884363030749882}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 6){
        poly_order = 5;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[5][6] = {
            {  0.117287442123209,   0.746024247475164,    1.55845949566284,    1.55845949566284,    0.746024247475164,    0.117287442123209},
            {  0.332881028694340,   0.874530715528338,    0.543560418955243,   -0.543560418955243,   -0.874530715528338,   -0.332881028694340},
            {  0.288094139387468,   0.124856100697577,   -0.447934271132893,   -0.447934271132893,    0.124856100697577,    0.288094139387468},
            {  0.0519392103185590, -0.163005111557524,   -0.149339402601609,    0.149339402601609,    0.163005111557524,   -0.0519392103185590},
            { -0.0362377464994744,   0,                    0,                    0,                    0,                   -0.0362377464994744}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 7){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][7] = {
            { 0.0949784168314860, 0.566018291318305, 1.31356667413863, 1.70008170491476, 1.31356667413863, 0.566018291318305, 0.0949784168314860 },
            { 0.250587094782525, 0.681489967749969, 0.690657445719605, 2.00000000000000e-17, -0.690657445719605, -0.681489967749969, -0.250587094782525 },
            { 0.202007643774510, 0.168629406848863, -0.190879017739020, -0.430288361398528, -0.190879017739020, 0.168629406848863, 0.202007643774510 },
            { 0.0416943617261924, -0.0742721289951722, -0.134644820503445, 8.00000000000000e-17, 0.134644820503445, 0.0742721289951722, -0.0416943617261924 },
            { -0.0168133620392770, -0.0318803294159158, 0.0105207408750771, 0.0455912688652087, 0.0105207408750771, -0.0318803294159158, -0.0168133620392770 },
            { -0.00696163036307614, 0, 0.0113689471456670, 0, -0.0113689471456670, 0, 0.00696163036307614 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 8){
        poly_order = 6;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[6][8] = {
            { 0.0800953311466389, 0.444613342434497, 1.08327139483605, 1.61939300666862, 1.61939300666862, 1.08327139483605, 0.444613342434497, 0.0800953311466389 },
            { 0.197609026471597, 0.534821958924058, 0.673438141686406, 0.316010377769360, -0.316010377769360, -0.673438141686406, -0.534821958924058, -0.197609026471597 },
            { 0.148389745304953, 0.156110582243925, -0.0474507685371932, -0.289652513686333, -0.289652513686333, -0.0474507685371932, 0.156110582243925, 0.148389745304953 },
            { 0.0305910886423633, -0.0333756722743367, -0.0898091723946003, -0.0503904085626158, 0.0503904085626158, 0.0898091723946003, 0.0333756722743367, -0.0305910886423633 },
            { -0.00858660181434638, -0.0197071657128553, -0.00409855066939471, 0.0222457738424189, 0.0222457738424189, -0.00409855066939471, -0.0197071657128553, -0.00858660181434638 },
            { -0.00365157403250102, 0, 0, 0, 0, 0, 0, 0.00365157403250102 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
       
       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       Fourier_spreading_coeff[0] = 1.61105318798171; Fourier_spreading_coeff[1] = 0.0808800424612325; Fourier_spreading_coeff[2] = -5.83030357103232; Fourier_spreading_coeff[3] = 2.75313980415827; Fourier_spreading_coeff[4] = 4.05989941037912; Fourier_spreading_coeff[5] = -2.66275733456109;
    }
    else if(spreading_closet == 0.05){
        spreading_select_c = 5.1136;
        spreading_Lambda_0 = 1.1081865356645;

        Fourier_spreading_order = 8;

        if(order == 2){
          poly_order = 8;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[8][2] = {
            {  0.882210748643093,   0.882210748643093  },
            {  2.13749612914007,   -2.13749612914007  },
            { -0.116217692294086,  -0.116217692294086 },
            { -3.01503898772574,    3.01503898772574  },
            { -0.721236722980455,  -0.721236722980455 },
            {  1.83740203019376,   -1.83740203019376  },
            {  0.483176745669771,   0.483176745669771  },
            { -0.581578028655538,   0.581578028655538  }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 3){
          poly_order = 7;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[7][3] = {
            {  0.535916990718485,   1.56031082210029,    0.535916990718485},
            {  1.29658038912763,   2.00000000000000e-17, -1.29658038912763},
            {  0.529923053101839,  -1.49059396181975,    0.529923053101839},
            { -0.615704315570463,   8.00000000000000e-17,  0.615704315570463},
            { -0.382784751185688,   0.533762763753863,   -0.382784751185688},
            {  0.118999156147980,   0,                   -0.118999156147980},
            {  0.0927592288234078,  0,                    0.0927592288234078}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 4){
          poly_order = 6;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[6][4] = {
            {  0.383237722248187,   1.36126704987023,    1.36126704987023,   0.383237722248187 },
            {  0.853759598411316,   0.753300926931140,  -0.753300926931140,  -0.853759598411316 },
            {  0.406097111860231,  -0.585809141260407,  -0.585809141260407,   0.406097111860231 },
            { -0.172536258874627,  -0.313650778372602,   0.313650778372602,   0.172536258874627 },
            { -0.131666453112603,   0.103627652368553,   0.103627652368553,  -0.131666453112603 },
            {  0.0112136745403109,  0.0524098939220667, -0.0524098939220667, -0.0112136745403109 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 5){
          poly_order = 5;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[5][5] = {
            {  0.302081162624797,   1.09167989296224,    1.56036469512603,   1.09167989296224,   0.302081162624797 },
            {  0.614233858453729,   0.808378518501759,   2.00000000000000e-17,  -0.808378518501759,  -0.614233858453729 },
            {  0.287898809191648,  -0.165601483801866,  -0.537893092090245,  -0.165601483801866,   0.287898809191648 },
            { -0.0595580391891401, -0.192970579883086,   8.00000000000000e-17,   0.192970579883086,   0.0595580391891401 },
            { -0.0561167599726182,  0,   0.0728020189742623,   0,  -0.0561167599726182 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 6){
          poly_order = 5;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[5][6] = {
            {  0.252925459487757,   0.882205240499574,   1.46917662032767,   1.46917662032767,   0.882205240499574,   0.252925459487757 },
            {  0.470948942440187,   0.712358389801192,   0.355862190114156,  -0.355862190114156,  -0.712358389801192,  -0.470948942440187 },
            {  0.208661573124563,  -0.0128562982542933,  -0.321197933063590,  -0.321197933063590,  -0.0128562982542933,   0.208661573124563 },
            { -0.0236292916770942, -0.109349236707158,  -0.0659277282748124,   0.0659277282748124,   0.109349236707158,   0.0236292916770942 },
            { -0.0271053246326607, -0.00879751378010571,   0.0286514577799173,   0.0286514577799173,  -0.00879751378010571,  -0.0271053246326607 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 7){
          poly_order = 5;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[5][7] = {
            {  0.220359727684056,   0.730012948044801,   1.30459075105055,   1.56036706984495,   1.30459075105055,   0.730012948044801,   0.220359727684056 },
            {  0.377840647115365,   0.602558512582053,   0.475414826005528,   2.00000000000000e-17,  -0.475414826005528,  -0.602558512582053,  -0.377840647115365 },
            {  0.156376324490107,   0.0409464876574374,  -0.168423706153106,  -0.274509712030554,  -0.168423706153106,   0.0409464876574374,   0.156376324490107 },
            { -0.00998361852261326, -0.0624380286030575,  -0.0626752775522970,   8.00000000000000e-17,   0.0626752775522970,   0.0624380286030575,   0.00998361852261326 },
            { -0.0144746680109203,  -0.00878834661761131,   0.00921438093572662,   0.0192194916866306,   0.00921438093572662,  -0.00878834661761131,  -0.0144746680109203 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }
        else if(order == 8){
          poly_order = 5;
          memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
          memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
          double array[5][8] = {
            {  0.197358496179726,   0.618868444188145,   1.14156257761575,   1.50852854806887,   1.50852854806887,   1.14156257761575,   0.618868444188145,   0.197358496179726 },
            {  0.313414188611971,   0.508200208016520,   0.490927941650304,   0.204520326168928,  -0.204520326168928,  -0.490927941650304,  -0.508200208016520,  -0.313414188611971 },
            {  0.120870681927417,   0.0570454675751950,  -0.0793577195213970,  -0.193350591500376,  -0.193350591500376,  -0.0793577195213970,   0.0570454675751950,   0.120870681927417 },
            { -0.00424777431505982, -0.0369152086985569,  -0.0474723927199542,  -0.0217164635356140,   0.0217164635356140,   0.0474723927199542,   0.0369152086985569,   0.00424777431505982 },
            { -0.00836236726555206,  -0.00660951960573490,   0,   0.0100669397902046,   0.0100669397902046,   0,  -0.00660951960573490,  -0.00836236726555206 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
        }

        memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
        double Fourier_array[8] = {1.72918334434787,	-0.000614502650135382,	-3.71526307417194,	-0.0794930663458162,	3.47945889936529,	-0.258925142986238,	-1.72028932888204,	0.644496940794116};
        for(int i=0; i<8; i++){
          Fourier_spreading_coeff[i] = Fourier_array[i];
        } 
    }
    else if(spreading_closet == 0.1){
       spreading_select_c = 4.2621;
       spreading_Lambda_0 = 1.21261853798378;
        
       Fourier_spreading_order = 4;
       
       if(order == 2){
        poly_order = 4;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        rho_coeff[0][0] = 0.941036732651868; rho_coeff[1][0] = 1.78611675908676; rho_coeff[2][0] = -0.519667829304986; rho_coeff[3][0] = -1.80146116443851;
        rho_coeff[0][1] = 0.941036732651868; rho_coeff[1][1] = -1.78611675908676; rho_coeff[2][1] = -0.519667829304986; rho_coeff[3][1] = 1.80146116443851;
       }
       else if(order == 3){
        poly_order = 4;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        rho_coeff[0][-1] = 0.63665296670533; rho_coeff[1][-1] = 1.20144802132803; rho_coeff[2][-1] = 0.176284697522182; rho_coeff[3][-1] = -0.502123109208998;
        rho_coeff[0][0] = 1.47816546036943; rho_coeff[1][0] = 2e-17; rho_coeff[2][0] = -1.0515576180115; rho_coeff[3][0] = 4e-17;
        rho_coeff[0][1] = 0.63665296670533; rho_coeff[1][1] = -1.20144802132803; rho_coeff[2][1] = 0.176284697522182; rho_coeff[3][1] = 0.502123109208998;
       }
       else if(order == 4){
        poly_order = 3;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");          
        rho_coeff[0][-1] = 0.489922352448191; rho_coeff[1][-1] = 0.810595531724709; rho_coeff[2][-1] = 0.206965409482158;
        rho_coeff[0][0] = 1.3273523413813; rho_coeff[1][0] = 0.551054738369726; rho_coeff[2][0] = -0.468526996456746;
        rho_coeff[0][1] = 1.3273523413813; rho_coeff[1][1] = -0.551054738369726; rho_coeff[2][1] = -0.468526996456746;
        rho_coeff[0][2] = 0.489922352448191; rho_coeff[1][2] = -0.810595531724709; rho_coeff[2][2] = 0.206965409482158;
       }
       else if(order == 5){
        poly_order = 3;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        rho_coeff[0][-2] = 0.407602107070562; rho_coeff[1][-2] = 0.620865471282995; rho_coeff[2][-2] = 0.168204757985836;          
        rho_coeff[0][-1] = 1.11399152893785; rho_coeff[1][-1] = 0.631836228551183; rho_coeff[2][-1] = -0.178671984948009;
        rho_coeff[0][0] = 1.48032600861502; rho_coeff[1][0] = 2.00000000000000e-17; rho_coeff[2][0] = -0.396930596275376;
        rho_coeff[0][1] = 1.11399152893785; rho_coeff[1][1] = -0.631836228551183; rho_coeff[2][1] = -0.178671984948009;
        rho_coeff[0][2] = 0.407602107070562; rho_coeff[1][2] = -0.620865471282995; rho_coeff[2][2] = 0.168204757985836;
       }
       else if(order == 6){
        poly_order = 3;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[3][6] = {
            {  0.355780172967334,   0.940884092946573,    1.41104519251124,    1.41104519251124,    0.940884092946573,    0.355780172967334},
            {  0.496513480902230,   0.587010871250854,    0.265377357420847,   -0.265377357420847,   -0.587010871250854,   -0.496513480902230},
            {  0.131158861121091,  -0.0571349666665080,  -0.247971028902638,   -0.247971028902638,   -0.0571349666665080,   0.131158861121091}
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 7){
        poly_order = 5;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[5][7] = {
            { 0.320365072583726, 0.810168339210813, 1.28361299971501, 1.48057203750656, 1.28361299971501, 0.810168339210813, 0.320365072583726 },
            { 0.414914052470860, 0.527286464910531, 0.372605319651315, 2.00000000000000e-17, -0.372605319651315, -0.527286464910531, -0.414914052470860 },
            { 0.105150756602291, -0.00623596841792556, -0.143180906832275, -0.205177269204465, -0.143180906832275, -0.00623596841792556, 0.105150756602291 },
            { -0.0221608921653160, -0.0468825205539005, -0.0384877921496588, 0, 0.0384877921496588, 0.0468825205539005, 0.0221608921653160 },
            { -0.00861247363531921, 0, 0, 0, 0, 0, -0.00861247363531921 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }
      else if(order == 8){
        poly_order = 4;
        memory->create2d_offset(rho_coeff,poly_order,(1-order)/2,order/2,"esp:rho_coeff");
        memory->create2d_offset(drho_coeff,poly_order,(1-order)/2,order/2,"esp:drho_coeff");
        double array[4][8] = {
            { 0.294888533003755, 0.711407535352690, 1.15430109120183, 1.44121465718693, 1.44121465718693, 1.15430109120183, 0.711407535352690, 0.294888533003755 },
            { 0.351330832004856, 0.458911465603961, 0.394611261309875, 0.155942673494966, -0.155942673494966, -0.394611261309875, -0.458911465603961, -0.351330832004856 },
            { 0.0822695471697131, 0.0150170895321342, -0.0797025067554486, -0.147913335360682, -0.147913335360682, -0.0797025067554486, 0.0150170895321342, 0.0822695471697131 },
            { -0.0134137715305704, -0.0297594094236832, -0.0303504704114467, -0.0127676126509343, 0.0127676126509343, 0.0303504704114467, 0.0297594094236832, 0.0134137715305704 }
          };
          for(int i=0; i<poly_order; i++){
              for(int j=0; j<order; j++){
                rho_coeff[i][j+(1-order)/2] = array[i][j];
              }
          }
      }

       memory->create(Fourier_spreading_coeff, Fourier_spreading_order, "esp:Fourier_spreading_coeff");
       Fourier_spreading_coeff[0] = 1.79345737218915; Fourier_spreading_coeff[1] = 0.102644452928885; Fourier_spreading_coeff[2] = -3.90688664859298; Fourier_spreading_coeff[3] = 2.18448520345598;
    }

    for (int m = -(order-1)/2; m <= order/2; m += 1) {
        for (int l = 1; l < poly_order; l++)
            drho_coeff[l-1][m] = l*rho_coeff[l][m]; // Coefficients for l x^l-1 terms
        drho_coeff[poly_order-1][m] = 0.00;    
    }
    return 0;
}


