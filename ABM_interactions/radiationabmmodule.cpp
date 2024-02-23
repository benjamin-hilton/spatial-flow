#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <random>
#include <iostream>
#include <chrono>
#include <algorithm> //for std::sort
#include <limits> //for infinity
#include <cmath>
#include <iomanip>
//May need to #include other things

#include <Python.h>
#include <numpy/arrayobject.h>

#include "radiationabmmodule.hpp"
#include "erf.hpp"

//Constructor
Radiation_ABM::Radiation_ABM(double std_dev_input,
                             double* population_list, unsigned population_list_size,
                             double* distance_matrix, unsigned distance_matrix_size,
                             double* job_list,
                             unsigned* flow_matrix_ptr)
     : m_std_dev(std_dev_input)
     , m_population_list(population_list)
     , m_distance_matrix(distance_matrix)
     , m_population_list_size(population_list_size)
     , m_distance_matrix_size(distance_matrix_size)
     , m_agent_list(population_list, population_list + population_list_size)
     , m_job_list(job_list, job_list + population_list_size)
     , m_flow_matrix(flow_matrix_ptr)
     , m_ascending_distance_matrix(std::vector<std::vector<std::pair<double, int>>>
            (population_list_size))
    , m_equidistant_slices(std::vector<std::vector<std::pair<unsigned, unsigned>>>
            (population_list_size))
{

    //TODO Create a list of agents at each location equal to the population
    //      at said location -  See above

    //Creates a 1D psuedo-matrix of pairs of the form <distance, index>
    //where the elements of ith row (of length m_population_list_size)
    //are the distances between site i and the site labelled by the index

    make_ascending_distance_matrix();

    find_equidistant_slices();

    create_randomised_agent_list();


}

//Destructor
Radiation_ABM::~Radiation_ABM()
{}


//General (Iterator) Methods

//Adds a new agent to a random sites

void Radiation_ABM::add_random_new_agent()
{
    //Seed based on time for Mersenne Twister
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();
    std::mt19937 generator{seed};

    //Maximum site index = (list length) - 1
    unsigned max = m_population_list_size - 1;

    //Using a uniform integer distribution to choose a site at randomadd_random
    std::uniform_int_distribution<unsigned> dist(0, max);

    unsigned random_site = dist(generator);

    // std::cout<<"Adding a random agent at site: "<<random_site<<"\n";

    m_agent_list[random_site]++;

}

double Radiation_ABM::gaussian_value(double mean, double sd)
{
    //Seed based on time for Mersenne Twister
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();
    std::mt19937 generator{seed};

    std::normal_distribution<double> normal_dist(mean, sd);

    double value = normal_dist(generator);

    // std::cout<<"Value from Gaussian dist: "<<value<<"\n";

    return value;
}

double Radiation_ABM::lognormal_value(double mu, double sigma)
{
    //Seed based on time for Mersenne Twister
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();
    std::mt19937 generator{seed};

    std::lognormal_distribution<double> lognormal_dist(mu, sigma);

    double value = lognormal_dist(generator);

    // std::cout<<"Value from lognormal dist: "<<value<<"\n";

    return value;
}

double Radiation_ABM::max_gaussian_value(double mean, double sd, unsigned n)
{
    //Begin by setting 'max_value' and 'new_value' to largest finite -ve doubles
    double max_value = -1 * std::numeric_limits<double>::max();
    double current_value = -1 * std::numeric_limits<double>::max();

    //std::cout<<"n = "<<n<<"\n";

    for(unsigned i = 0; i < n; i++)
    {
        current_value = gaussian_value(mean, sd);
        if(current_value >= max_value) max_value = current_value;
    }

    return max_value;
}

double Radiation_ABM::max_lognormal_value(double mu, double sigma, unsigned n)
{
    // //Begin by setting 'max_value' and 'new_value' to largest finite -ve doubles
    // double max_value = -1 * std::numeric_limits<double>::max();
    // double current_value = -1 * std::numeric_limits<double>::max();
    //
    // //std::cout<<"n = "<<n<<"\n";
    //
    // for(unsigned i = 0; i < n; i++)
    // {
    //     current_value = lognormal_value(mu, sigma);
    //     if(current_value >= max_value) max_value = current_value;
    // }
    //
    // return max_value;

    //Using record stats approach

    //Seed based on time for Mersenne Twister
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();
    std::mt19937 generator{seed};

    std::uniform_real_distribution<long double> uniform(0.0, 1.0);

    long double x = std::pow(uniform(generator), 1.0/n);

    //std::cout <<  std::setprecision (17) <<  x << ' ' << n << ' ' << 1/n << std::endl;

    double result = std::exp(mu + sigma * M_SQRT2 * (erf_hpp::erfinv<long double>(2 * x - 1)));

    //std::cout<<result<<std::endl;

    return result;

}

void Radiation_ABM::create_randomised_agent_list()
{
    //Find total pop
    //cast total_population to unsigned
    //declare size of m_agent_list
    //fill m_agent_list with {0,0,0,...1,1,1... etc.}


    double d_total_population = 0.;

    for(unsigned j = 0; j < m_population_list_size; j++)
    {
        d_total_population += m_population_list[j];
    }

    m_total_population = static_cast<unsigned> (d_total_population);

    //Create a vector of the right size
    //Populate it with appropriate elements
    //Shuffle these elements
    //Assign that vector to m_agent_list
    //TODO think of a more efficient way of doing this
    std::vector<unsigned> vec(m_total_population);

    unsigned n = 0;
    for(unsigned i = 0; i < m_population_list_size; i++)
    {
        for(unsigned j = 0; j < m_population_list[i]; j++)
        {
            vec[n] = i;
            n++;
        }

    }

    //TODO change to reference?
    m_agent_list = vec;

    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();

    //TODO SHUFFLE
    std::shuffle(m_agent_list.begin(), m_agent_list.end(),
    std::default_random_engine(seed));

}


std::vector<std::pair<double, int>> Radiation_ABM::pair_with_indices(const std::vector<double> &vec)
{

    std::vector<std::pair<double, int>> pair_vector(vec.size());

    for(unsigned i = 0; i < vec.size(); i++)
    {
        pair_vector[i] = std::make_pair(vec[i], i);
    }

    return pair_vector;
}

void Radiation_ABM::make_ascending_distance_matrix()
{

    for(unsigned i = 0; i < m_population_list_size; i++)
    {
        //Split into rows and make pairs

        //Select a "row" from the pseudo matrix
        std::vector<double> distance_matrix_row
        (m_distance_matrix + i * m_population_list_size
            , m_distance_matrix + (i+1) * m_population_list_size);

        //Turn said row of distances into a row of pairs where
        //each value is paired with its index
        //(elements of the form <dist0, 0>, <dist1, 1> and so on)
        std::vector<std::pair<double, int>> pair_row
                                        = pair_with_indices(distance_matrix_row);

        //Sort row of pairs so it's organised in ascending order by distance
        std::sort(pair_row.begin(), pair_row.end());

        m_ascending_distance_matrix[i] = pair_row;
    }

    //Shuffle any equidistant sites to ensure randomness in selection

    //Could shuffle once but only actually need to do it at the end since
    //each "row" should start with 0 (considering the self-loop)

    //shuffle_equidistant_pairs(m_ascending_distance_matrix);

}

void Radiation_ABM::shuffle_equidistant_pairs(std::vector<std::pair<double, int>> &pair_vector)
{
    unsigned j;

    //Assign a random seed for std::shuffle
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();

    //Iterate through vector of pairs
    for(unsigned i = 0; i < pair_vector.size(); i++)
    {
        //If any of the elements have the same first value...
        if(pair_vector[i].first == pair_vector[i+1].first)
        {
            j = i + 1;

            //...check all subsequent elements
            //until all those with the same first value are accounted for
            while(pair_vector[j].first == pair_vector[i].first)
            {
                j++;
            }

            std::shuffle(pair_vector.begin()+i, pair_vector.begin()+j
                        , std::default_random_engine(seed));


            i = j - 1;

        }
    }


}

std::vector<std::pair<unsigned, unsigned>>
    Radiation_ABM::find_equal_adjacent_pairs(
        const std::vector<std::pair<double, int>> &pair_vector)
{
    std::vector<std::pair<unsigned, unsigned>> equal_value_slices;
    std::pair<unsigned, unsigned> slice;

    unsigned j;

    //Iterate through vector of pairs
    for(unsigned i = 0; i < pair_vector.size(); i++)
    {
        //If any of the elements have the same first value...
        if(pair_vector[i].first == pair_vector[i+1].first)
        {
            j = i + 1;

            //...check all subsequent elements
            //until all those with the same first value are accounted for
            while(pair_vector[j].first == pair_vector[i].first)
            {
                j++;
            }

            slice = std::make_pair(i, j - 1);

            equal_value_slices.push_back(slice);

            i = j;

        }

    }

    return equal_value_slices;
}

void Radiation_ABM::find_equidistant_slices()
{
    for(unsigned i = 0; i < m_population_list_size; i++)
    {
        m_equidistant_slices[i] =
            find_equal_adjacent_pairs(m_ascending_distance_matrix[i]);
    }

}

void Radiation_ABM::shuffle_equidistant_sites(unsigned home_site)
{

  //Assign a random seed for std::shuffle
  std::random_device rd;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rd();

  for(unsigned j = 0; j < m_equidistant_slices[home_site].size(); j++)
  {
        //Shuffle equidistant elements
        //Specified by pairs in m_equidistant_slices
        //on the interval [first,second)
        std::shuffle(m_ascending_distance_matrix[home_site].begin() + m_equidistant_slices[home_site][j].first,
                     m_ascending_distance_matrix[home_site].begin() + m_equidistant_slices[home_site][j].second+1,
                    std::default_random_engine(seed));

  }

}

void Radiation_ABM::run()
{

    double mean = 33419; //set to mean US income //TODO Change for other datasets
    double sd = m_std_dev;
    double var = sd * sd;

    double mu = log(mean * mean / sqrt(var + mean * mean));
    double sigma = sqrt(log(var/(mean * mean) + 1));

    std::cout << "Mean:" << mean << std::endl;
    std::cout << "Standard deviation:" << sd << std::endl;
    std::cout << "mu:" << mu << std::endl;
    std::cout << "sigma:" << sigma << std::endl;

    double local_z, target_z;
    unsigned i = 1;

    for(unsigned n = 0; n < m_total_population; n++)
    {
        unsigned home_site = m_agent_list[n];
        unsigned target_site = home_site;

        //assign local_z using population
        local_z = max_lognormal_value(mu, sigma, m_population_list[home_site]);

        shuffle_equidistant_sites(home_site);

        do{
          //To check each element of m_ascending_distance_matrix row
          //NB: Not the same as site index due to rearrangement/shuffling
          //site index is second term in each pair
          //Starting from 1 as 0th element will be home_site

          target_site = m_ascending_distance_matrix[home_site][i].second;
          //std::cout<<"Target site: "<<target_site<<"\n";

          if(m_job_list[target_site] > 0)
          {

              //assign target_z using jobs
              target_z = max_lognormal_value(mu, sigma,
              m_job_list[target_site]);

              //std::cout<<"i = "<<i<<"\n";

              //std::cout<<"Target value: "<<target_z<<"\n";

              //If site with higher z found...
              if(local_z <= target_z)
              {
                  //std::cout<<"Increment ["<<home_site<<"]["<<target_site<<"]\n";
                  //..update corresponding element in flow matrix
                  m_flow_matrix[home_site*m_population_list_size + target_site] += 1;
                  m_job_list[target_site]--;
              }

          }

          //If not, check next site
          i++;

        } while(local_z > target_z && i < m_population_list_size);

        i = 1; //reset index

    }



}

//Accessor methods
const double* Radiation_ABM::get_population_list()
{
    return m_population_list;
}

const double* Radiation_ABM::get_distance_matrix()
{
    return m_distance_matrix;
}

unsigned Radiation_ABM::get_population_list_size()
{
    return m_population_list_size;
}

unsigned Radiation_ABM::get_distance_matrix_size()
{
    return m_distance_matrix_size;
}

std::vector<unsigned> Radiation_ABM::get_agent_list()
{
    return m_agent_list;
}

unsigned* Radiation_ABM::get_flow_matrix()
{
    return m_flow_matrix;
}

std::vector<std::vector<std::pair<double, int>>> Radiation_ABM::get_ascending_distance_matrix()
{
    return m_ascending_distance_matrix;
}

std::vector<std::vector<std::pair<unsigned, unsigned>>> Radiation_ABM::get_equidistant_slices()
{
    return m_equidistant_slices;
}

static PyObject* run_abm(PyObject *self, PyObject *args) {

    // Create PyObject into which data is parsed.
    PyObject* population_list = NULL;
    PyObject* distance_matrix = NULL;
    PyObject* job_list = NULL;
    double std_dev;

    // Parse the arguments into C++ variables and check for success.
    if (!PyArg_ParseTuple(args, "OOOd", &population_list, &distance_matrix, &job_list, &std_dev)) {
        // Returning NULL tells Python there's been an error.
        return NULL;
    }
    if (population_list == NULL || distance_matrix == NULL || job_list == NULL) {
        return NULL;
    }

    // Check objects are arrays.
    if (!PyArray_Check(population_list)) {
        PyErr_SetString(PyExc_TypeError, "Input population_list is not a NumPy array.");
        return NULL;
    }
    if (!PyArray_Check(distance_matrix)) {
        PyErr_SetString(PyExc_TypeError, "Input distance_matrix is not a NumPy array.");
        return NULL;
    }
    if (!PyArray_Check(job_list)) {
        PyErr_SetString(PyExc_TypeError, "Input job_list is not a NumPy array.");
        return NULL;
    }

    // Check the data type of the arrays.
    if (PyArray_TYPE((PyArrayObject*)population_list) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of population_list is not float64.", 1);
    }
    if (PyArray_TYPE((PyArrayObject*)distance_matrix) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of distance_matrix is not float64.", 1);
    }
    if (PyArray_TYPE((PyArrayObject*)job_list) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of job_list is not float64.", 1);
    }

    // Check that the arrays are C contiguous.
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)population_list)) {
        PyErr_SetString(PyExc_ValueError, "Array population_list is not C contiguous.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)distance_matrix)) {
        PyErr_SetString(PyExc_ValueError, "Array distance_matrix is not C contiguous.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)job_list)) {
        PyErr_SetString(PyExc_ValueError, "Array job_list is not C contiguous.");
        return NULL;
    }

    // Check that the dimensions of the input arrays are correct.
    if (PyArray_NDIM((PyArrayObject*)population_list) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array population_list must be 1-dimensional.");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject*)job_list) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array job_list must be 1-dimensional.");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject*)distance_matrix) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array distance_matrix must be 2-dimensional.");
        return NULL;
    }

    // Check that the standard deviation is greater than zero.
    if (std_dev <= 0) {
        PyErr_SetString(PyExc_ValueError, "Standard deviation must be greater than zero.");
        return NULL;
    }

    const unsigned long distance_size = PyArray_Size(distance_matrix);
    const unsigned long population_size = PyArray_Size(population_list);
    const unsigned long job_size = PyArray_Size(job_list);

    // Check that the size of distance_matrix is equal to the square of the size of input_list.
    if (population_size * population_size != distance_size) {
        PyErr_SetString(PyExc_ValueError, "The distance_matrix array must be a square matrix with rows of size equal to the size of population_list.");
        return NULL;
    }

    // Check that size of population_list is equal to the size of job_list.
    if (population_size != job_size) {
        PyErr_SetString(PyExc_ValueError, "The job_list array must be the same size as the population_list array.");
        return NULL;
    }

    // Create pointers to access array data.
    double* population_list_ptr = (double *)PyArray_GETPTR1((PyArrayObject*)population_list, 0);
    double* job_list_ptr = (double *)PyArray_GETPTR1((PyArrayObject*)job_list, 0);
    double* distance_matrix_ptr = (double *)PyArray_GETPTR2((PyArrayObject*)distance_matrix, 0, 0);

    npy_intp* distance_matrix_dims = PyArray_DIMS((PyArrayObject*)distance_matrix);
    PyObject* flow_matrix_ndarray = (PyObject*)PyArray_SimpleNew(2, distance_matrix_dims, NPY_UINT32);
    unsigned* flow_matrix_ptr = (unsigned*)PyArray_GETPTR2((PyArrayObject*)flow_matrix_ndarray, 0, 0);

    for(unsigned i = 0; i < population_size; i++)
    {
        for(unsigned j = 0; j < population_size; j++)
        {
            flow_matrix_ptr[i*population_size + j] = 0;
        }
    }

    Radiation_ABM iterator(std_dev, population_list_ptr, population_size, distance_matrix_ptr, distance_size, job_list_ptr, flow_matrix_ptr);

    iterator.run();
    std::cout<<"DONE!\n";

    return flow_matrix_ndarray;

}

// Define a list of methods to be accessed by Python.
static PyMethodDef methods[] = {
    {"run_abm", (PyCFunction)run_abm, METH_VARARGS, run_abm__doc__},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// Define the module as a whole.
PyDoc_STRVAR(
    module__doc__,
    "Implements a radiation model ABM in C++ with interactions.\n"
);
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "radiationabm",
    module__doc__,
    -1,
    methods
 };

// Define the initialisation function for the module.
PyMODINIT_FUNC PyInit_radiationabm(void){
    import_array();
    return PyModule_Create(&module);
}
