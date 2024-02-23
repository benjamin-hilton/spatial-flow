/*
Created by
Abhijay Sood
17/01/2019
*/

#ifndef RADIATION_ABM_HPP
#define RADIATION_ABM_HPP

#include <vector>
#include <utility> //for std::pair
//May need to #include other things


//TODO Include Python docstrings of the form
// PyDoc_STRVAR(
//     function_name__doc__,
//     "function_name(*args, **kwargs)\n"
//     "--\n"
//     "\n"
//     "Description\n"
//     "\n"
//     "Arguments\n"
//     "----------\n"
//     "    argument: description\n"
//     "Returns\n"
//     "----------\n"
//     "    thing: description"

class Radiation_ABM
{
private:

    //Marked const as these should not be changed by program

    const double m_std_dev;

    //List of populations of each site
    const double* m_population_list;

    //flattened 1D pseudo-matrix
    //should have size = population list size **2
    const double* m_distance_matrix;

    //Number of elements in each of the above vectors
    const unsigned m_population_list_size;
    const unsigned m_distance_matrix_size;

    //TODO Const-ify this?
    //Sum population over all sites
    unsigned m_total_population;

    //Vector of all agents
    //numbered by their home site
    std::vector<unsigned> m_agent_list;

    //Vector of the number of jobs at each site;
    //size == m_population_list_size
    std::vector<unsigned> m_job_list;


    //Vector of the flows between each site;
    //size == m_distance_matrix_size
    unsigned* m_flow_matrix;
    //TODO Change to double?

    //Vector of vectors
    //Entries in each row are 'paired' with their index in said row and then
    //arranged in ascending order (by value)
    std::vector<std::vector<std::pair<double, int>>> m_ascending_distance_matrix;

    //List of slices in ascending distance matrix that represent
    //sets of equidistant sites so the shuffle algorithm needn't search
    //through the matrix to find them every time
    //The ith element contains a list of all the slices of sites that are
    //equidistant from site i

    std::vector<std::vector<std::pair<unsigned, unsigned>>> m_equidistant_slices;

public:

    //Constructor - const private member variables will req initialisation list
    Radiation_ABM(double std_dev_input,
                  double* population_list, unsigned population_list_size,
                  double* distance_matrix, unsigned distance_matrix_size,
                  double* job_list,
                  unsigned* flow_matrix_ptr);

    //Destructor
    ~Radiation_ABM();

    //General (Iterator) methods

        //Adds a new agent to a random site
        void add_random_new_agent();

        //Returns a value chosen from a Gaussian distribution
        double gaussian_value(double mean, double sd);

        //Returns maximum value after n draws from Gaussian distribution
        double max_gaussian_value(double mean, double sd, unsigned n);

        //Returns a value chosen from a lognormal distribution
        double lognormal_value(double mu, double sigma);

        //Returns maximum value after n draws from lognormal distribution
        double max_lognormal_value(double mu, double sigma, unsigned n);

        //So that agents are sent out in a random order
        void create_randomised_agent_list();


        //For the ordered distance matrix:

            //"Pairs" elements of a vector of doubles with its index in said vector
            std::vector<std::pair<double, int>> pair_with_indices(const std::vector<double> &vec);

            //Creates a flattened 1D pseudo-matrix
            //Entries in each row are 'paired' with their index in said row and then
            //arranged in ascending order
            void make_ascending_distance_matrix();

            //Shuffles equidistant pairs, to make sure they are chosen at random
            //e.g. The order of the middle three elements listed here may be swapped
            // <0, 3> <5, 1> <5, 2> <5,4> <7, 0>
            //TODO delete this, I don't need it anymore
            void shuffle_equidistant_pairs(std::vector<std::pair<double, int>> &pair_vector);

            //Need to SAVE EQUIDISTANT SLICES
            //SHUFFLE ACCORDING TO THESE SLICES

            //NEED TO INITIALISE m_equidistant_slices with size in initialisation list


            //For a vector of pairs of the form
            //(<double, int>, <double, int> ... , <double, int>)
            //Returns locations of adjacent elements with equal double values
            //i.e. equal values in the first element
            //For each set of equal value elements, the locations are returned as
            //a pair with the indices of the <first, last> elements
            //Meaning the output is a vector of pairs where each element
            //gives the indices of elements between which distances are equal
            std::vector<std::pair<unsigned, unsigned>>
                find_equal_adjacent_pairs(const std::vector<std::pair<double, int>> &pair_vector);

            //Using find_equal_adjacent_pairs, creates a
            //list of slices in ascending distance matrix that represent
            //sets of equidistant sites so the shuffle algorithm needn't search
            //through the matrix to find them every time
            //The ith element contains a list of all the slices of sites that are
            //equidistant from site i
            void find_equidistant_slices();

            //Shuffles equidistant sites
            //Relative to location of given home_site
            //TODO Add a caveat that means this doesn't have to be engaged
            //in run() IF there are no equidistant sites
            void shuffle_equidistant_sites(unsigned home_site);

        //Moves agents following to assumptions in SGMB paper (2012),
        //updates flow matrix accordingly
        void run();


    //Accessor Methods
        const double* get_population_list();
        const double* get_distance_matrix();

        unsigned get_population_list_size();
        unsigned get_distance_matrix_size();

        std::vector<unsigned> get_agent_list();

        unsigned* get_flow_matrix();

        std::vector<std::vector<std::pair<double, int>>> get_ascending_distance_matrix();

        std::vector<std::vector<std::pair<unsigned, unsigned>>> get_equidistant_slices();


};


PyDoc_STRVAR(
    run_abm__doc__,
    "run_abm(population_list, distance_matrix)\n"
    "--\n"
    "\n"
    "Calculates radiation model values using an ABM.\n"
    "\n"
    "Arguments\n"
    "----------\n"
    "    population_list: A 1-D NumPy float array containing populations at\n"
    "        each site.\n"
    "    distance_matrix: A 2-D square NumPy float array of size equal to the\n"
    "        square of the size of population_list, where distance_matrix[i][j] is\n"
    "        the distance between site i (with population population_list[i]) and\n"
    "        site j (with population population_list[j]).\n"
    "    std_dev: The standard deviation of the lognormal distribution."
    "\n"
    "Returns\n"
    "----------\n"
    "    values: A 2-D square NumPy array containing the radiation model values\n"
    "    between every pair of points. The value in values[i][j] corresponds to the\n"
    "    distance in distance_matrix[i][j].\n"
);
static PyObject* run_abm(PyObject *self, PyObject *args);


#endif // RADIATION_ABM_HPP
