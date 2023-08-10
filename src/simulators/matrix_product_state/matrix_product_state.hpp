/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

//=========================================================================
// Tensor Network State - simulation method
//=========================================================================
// For this simulation method, we represent the state of the circuit using a
// tensor network structure, the specifically matrix product state. The idea is
// based on the following paper (there exist other sources as well): The
// density-matrix renormalization group in the age of matrix product states by
// Ulrich Schollwock.
//
//--------------------------------------------------------------------------

#ifndef _matrix_product_state_hpp
#define _matrix_product_state_hpp

#include <algorithm>
#include <sstream>
#define _USE_MATH_DEFINES
#define MAX_PARALLEL_OPS 4
#include <math.h>

#include "framework/json.hpp"
#include "framework/linalg/almost_equal.hpp"
#include "framework/utils.hpp"
#include "matrix_product_state_internal.cpp"
#include "matrix_product_state_internal.hpp"
#include "simulators/state.hpp"

namespace AER {
namespace MatrixProductState {

static uint_t instruction_number = 0;

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    {OpType::gate,
     OpType::measure,
     OpType::reset,
     OpType::initialize,
     OpType::barrier,
     OpType::bfunc,
     OpType::roerror,
     OpType::qerror_loc,
     OpType::matrix,
     OpType::diagonal_matrix,
     OpType::kraus,
     OpType::save_expval,
     OpType::save_expval_var,
     OpType::save_densmat,
     OpType::save_statevec,
     OpType::save_probs,
     OpType::save_probs_ket,
     OpType::save_amps,
     OpType::save_amps_sq,
     OpType::save_mps,
     OpType::save_state,
     OpType::set_mps,
     OpType::set_statevec,
     OpType::jump,
     OpType::mark},
    // Gates
    {"id",  "x",    "y",   "z",   "s",     "sdg",   "h",    "t",  "tdg", "p",
     "u1",  "u2",   "u3",  "u",   "U",     "CX",    "cx",   "cy", "cz",  "cp",
     "cu1", "swap", "ccx", "sx",  "sxdg",  "r",     "rx",   "ry", "rz",  "rxx",
     "ryy", "rzz",  "rzx", "csx", "delay", "cswap", "pauli"});

//=========================================================================
// Matrix Product State subclass
//=========================================================================

using matrixproductstate_t = MPS;
using OpItr = std::vector<Operations::Op>::const_iterator;

class State : public QuantumState::State<matrixproductstate_t> {
public:
  using BaseState = QuantumState::State<matrixproductstate_t>;

  State() : BaseState(StateOpSet) {}
  State(uint_t num_qubits) : State() { qreg_.initialize((uint_t)num_qubits); }
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "matrix_product_state"; }

  bool empty() const { return qreg_.empty(); }

  // Iterate over the operations to apply and create subsets of first and last
  // operation to execute in parallel
  std::pair<OpItr, OpItr> identify_parallel_ops(const OpItr first, const OpItr last, bool &can_parallel) {

    std::pair<OpItr, OpItr> parallel_ops;
    OpItr tmp_first = first, tmp_last = first;
    bool overlap = false;
    uint_t q0, q1, t_q0, t_q1, aux0, aux1;

    for (auto it = first+1; it != last; ++it) {
      if (it->qubits.size() <= 2 and (it->type == OpType::matrix or it->type == OpType::gate)) {
        if (it->qubits.size() == 1) {
          q0 = qreg_.qubit_ordering_.location_[it->qubits[0]];
        }
        else {
          aux0 = qreg_.qubit_ordering_.location_[it->qubits[0]];
          aux1 = qreg_.qubit_ordering_.location_[it->qubits[1]];
          q0 = std::min(aux0, aux1);
          q1 = std::max(aux0, aux1);
        }

        // compare with previous operations to see if current is compatible
        for (auto prev_it = it-1; prev_it >= tmp_first; --prev_it) {
          if (prev_it->qubits.size() == 1) {
              t_q0 = qreg_.qubit_ordering_.location_[prev_it->qubits[0]];
              if (it->qubits.size() == 1 and t_q0 == q0) {
                  overlap = true;
                  break;
              } else if (it->qubits.size() == 2 and (t_q0 <= q1 and t_q0 >= q0)) {
                  overlap = true;
                  break;
              }
          }
          else if (prev_it->qubits.size() == 2) {
            // qubits do not have to be ordered from lower to higher
            aux0 = qreg_.qubit_ordering_.location_[prev_it->qubits[0]];
            aux1 = qreg_.qubit_ordering_.location_[prev_it->qubits[1]];
            t_q0 = std::min(aux0, aux1);
            t_q1 = std::max(aux0, aux1);

            // negate of (qubit0 > t_qubit1 || qubit1 < t_qubit0) == no overlap
            if (it->qubits.size() == 1 and (q0 <= t_q1 and q0 >= t_q0)) {
              // here qubits are always ordered pos0=lower, pos1=higher -> (0,4), (1,2), (1,1), etc.
              overlap = true;
              break;
            } else if (it->qubits.size() == 2 and (q0 <= t_q1 and q1 >= t_q0)) {
              overlap = true;
              break;
            }
          }
        }
        if (overlap) {
            break;
        }
        tmp_last = it;
      } else {
        break;
      }
    }
    if (tmp_first != tmp_last) {
      // if different, there must be parallel operations
      can_parallel = true;
      parallel_ops = std::pair<OpItr, OpItr>(tmp_first, tmp_last);
    }

    return parallel_ops;
  }

  virtual void apply_ops(const OpItr first, const OpItr last,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false) override;

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // We currently set the threshold to 1 in qasm_controller.hpp, i.e., no
  // parallelization
  virtual void set_config(const Config &config) override;

  virtual void add_metadata(ExperimentResult &result) const override;

  // prints the bond dimensions after each instruction to the metadata
  void output_bond_dimensions(const Operations::Op &op) const;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) override;

  // Computes sample_measure by copying the MPS to a temporary structure, and
  // applying a measurement on the temporary MPS. This is done for every shot,
  // so is not efficient for a large number of shots
  std::vector<reg_t> sample_measure_using_apply_measure(const reg_t &qubits,
                                                        uint_t shots,
                                                        RngEngine &rng);
  std::vector<reg_t> sample_measure_all(uint_t shots, RngEngine &rng);
  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  void initialize_omp();

protected:
  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const Operations::Op &op);

  // Initialize the specified qubits to a given state |psi>
  // by creating the MPS state with the new state |psi>.
  // |psi> is given in params
  // Currently only supports intialization of all qubits
  void apply_initialize(const reg_t &qubits, const cvector_t &params,
                        RngEngine &rng);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function, then "measure"
  // should be contained in the set defined by 'allowed_ops'
  virtual void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Apply a matrix to given qubits (identity on all other qubits)
  // We assume matrix to be 2x2
  void apply_matrix(const reg_t &qubits, const cmatrix_t &mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t &vmat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng);

  // Apply multi-qubit Pauli
  void apply_pauli(const reg_t &qubits, const std::string &pauli);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the simulator
  void apply_save_mps(const Operations::Op &op, ExperimentResult &result,
                      bool last_op);

  // Compute and save the statevector for the current simulator state
  void apply_save_statevector(const Operations::Op &op,
                              ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function, then "measure"
  // must be contained in the set defined by 'allowed_ops'
  rvector_t measure_probs(const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_with_prob(const reg_t &qubits,
                                                     RngEngine &rng);

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Apply a waltz gate specified by parameters u3(theta, phi, lambda)
  void apply_gate_u3(const uint_t qubit, const double theta, const double phi,
                     const double lambda);

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;
};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

const stringmap_t<Gates>
    State::gateset_({                   // Single qubit gates
                     {"id", Gates::id}, // Pauli-Identity gate
                     {"delay", Gates::id},
                     {"x", Gates::x},       // Pauli-X gate
                     {"y", Gates::y},       // Pauli-Y gate
                     {"z", Gates::z},       // Pauli-Z gate
                     {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
                     {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
                     {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
                     {"sx", Gates::sx},     // Sqrt(X) gate
                     {"sxdg", Gates::sxdg}, // Inverse Sqrt(X) gate
                     {"t", Gates::t},       // T-gate (sqrt(S))
                     {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
                     {"r", Gates::r},       // R rotation gate
                     {"rx", Gates::rx},     // Pauli-X rotation gate
                     {"ry", Gates::ry},     // Pauli-Y rotation gate
                     {"rz", Gates::rz},     // Pauli-Z rotation gate
                     /* Waltz Gates */
                     {"p", Gates::u1},  // zero-X90 pulse waltz gate
                     {"u1", Gates::u1}, // zero-X90 pulse waltz gate
                     {"u2", Gates::u2}, // single-X90 pulse waltz gate
                     {"u3", Gates::u3}, // two X90 pulse waltz gate
                     {"u", Gates::u3},  // two X90 pulse waltz gate
                     {"U", Gates::u3},  // two X90 pulse waltz gate
                     /* Two-qubit gates */
                     {"CX", Gates::cx},   // Controlled-X gate (CNOT)
                     {"cx", Gates::cx},   // Controlled-X gate (CNOT)
                     {"cy", Gates::cy},   // Controlled-Y gate
                     {"cz", Gates::cz},   // Controlled-Z gate
                     {"cu1", Gates::cu1}, // Controlled-U1 gate
                     {"cp", Gates::cu1},  // Controlled-U1 gate
                     {"csx", Gates::csx},
                     {"swap", Gates::swap}, // SWAP gate
                     {"rxx", Gates::rxx},   // Pauli-XX rotation gate
                     {"ryy", Gates::ryy},   // Pauli-YY rotation gate
                     {"rzz", Gates::rzz},   // Pauli-ZZ rotation gate
                     {"rzx", Gates::rzx},   // Pauli-ZX rotation gate
                     /* Three-qubit gates */
                     {"ccx", Gates::ccx}, // Controlled-CX gate (Toffoli)
                     {"cswap", Gates::cswap},
                     /* Pauli */
                     {"pauli", Gates::pauli}});

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

void State::initialize_qreg(uint_t num_qubits = 0) {
  qreg_.initialize(num_qubits);
}

void State::initialize_omp() {
  if (BaseState::threads_ > 0)
    qreg_.set_omp_threads(
        BaseState::threads_); // set allowed OMP threads in MPS
}

size_t State::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops) const {
  // for each qubit we have a tensor structure.
  // Initially, each tensor contains 2 matrices with a single complex double
  // Depending on the number of 2-qubit gates,
  // these matrices may double their size
  // for now - compute only initial size
  // later - FIXME
  size_t mem_mb = 16 * 2 * num_qubits;
  return mem_mb;
}

void State::set_config(const Config &config) {
  // Set threshold for truncating Schmidt coefficients
  MPS_Tensor::set_truncation_threshold(
      config.matrix_product_state_truncation_threshold);

  if (config.matrix_product_state_max_bond_dimension.has_value())
    MPS_Tensor::set_max_bond_dimension(
        config.matrix_product_state_max_bond_dimension.value());
  else
    MPS_Tensor::set_max_bond_dimension(UINT64_MAX);

  // Set threshold for truncating snapshots
  MPS::set_json_chop_threshold(config.chop_threshold);

  // Set OMP num threshold
  MPS::set_omp_threshold(config.mps_parallel_threshold);

  // Set OMP threads
  MPS::set_omp_threads(config.mps_omp_threads);

  // Set the algorithm for sample measure
  if (config.mps_sample_measure_algorithm.compare("mps_probabilities") == 0)
    MPS::set_sample_measure_alg(Sample_measure_alg::PROB);
  else
    MPS::set_sample_measure_alg(Sample_measure_alg::APPLY_MEASURE);

  // Set mps_log_data
  MPS::set_mps_log_data(config.mps_log_data);

  // Set the direction for the internal swaps
  std::string direction;
  if (config.mps_swap_direction.compare("mps_swap_right") == 0)
    MPS::set_mps_swap_direction(MPS_swap_direction::SWAP_RIGHT);
  else
    MPS::set_mps_swap_direction(MPS_swap_direction::SWAP_LEFT);
}

void State::add_metadata(ExperimentResult &result) const {
  result.metadata.add(MPS_Tensor::get_truncation_threshold(),
                      "matrix_product_state_truncation_threshold");
  result.metadata.add(MPS_Tensor::get_max_bond_dimension(),
                      "matrix_product_state_max_bond_dimension");
  result.metadata.add(MPS::get_sample_measure_alg(),
                      "matrix_product_state_sample_measure_algorithm");
  if (MPS::get_mps_log_data())
    result.metadata.add("{" + MPS::output_log() + "}", "MPS_log_data");
}

void State::output_bond_dimensions(const Operations::Op &op) const {
  MPS::print_to_log("I", instruction_number, ":", op.name, " on qubits ",
                    op.qubits[0]);
  for (uint_t index = 1; index < op.qubits.size(); index++) {
    MPS::print_to_log(",", op.qubits[index]);
  }
  qreg_.print_bond_dimensions();
  instruction_number++;
}

/****
 * Overwrite default apply_ops to take into account the parallel operations case
****/
void State::apply_ops(const OpItr first, const OpItr last,
                          ExperimentResult &result, RngEngine &rng, bool final_ops) {

  OpItr first_p_op, last_p_op;
  std::pair<OpItr, OpItr> parallel_ops;
  bool do_parallel = false, can_parallel = false;
  std::unordered_map<std::string, OpItr> marks;
  int num_parallel_ops, save;

  if (getenv("QISKIT_MPS_PARALLEL_OPS")) {
    result.metadata.add(true, "matrix_product_state_parallel_ops");
    do_parallel = true;
  }

  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
      case Operations::OpType::mark: {
        marks[it->string_params[0]] = it;
        break;
      }
      case Operations::OpType::jump: {
        if (creg().check_conditional(*it)) {
          const auto& mark_name = it->string_params[0];
          auto mark_it = marks.find(mark_name);
          if (mark_it != marks.end()) {
            it = mark_it->second;
          } else {
            for (++it; it != last; ++it) {
              if (it->type == Operations::OpType::mark) {
                marks[it->string_params[0]] = it;
                if (it->string_params[0] == mark_name) {
                  break;
                }
              }
            }
            if (it == last) {
              std::stringstream msg;
              msg << "Invalid jump destination:\"" << mark_name << "\"." << std::endl;
              throw std::runtime_error(msg.str());
            }
          }
        }
        break;
      }
      default: {
        // For each operation, evaluate if it, and next operations
        // can be executed in parallel
        if (do_parallel and it != last and qreg_.num_qubits() > 8) {
          parallel_ops = identify_parallel_ops(it, last, can_parallel);
          if (can_parallel) {  // it is easier than work with the pair
            int th_num = 1;
            can_parallel = false;
            first_p_op = std::get<0>(parallel_ops);
            last_p_op = std::get<1>(parallel_ops);

            // Execute the following operations in parallel and update values
            num_parallel_ops = (int)(last_p_op - first_p_op)+1;
            if (num_parallel_ops >= 2) {
                // Probably this will always be 2 or 1
              th_num = std::min(MAX_PARALLEL_OPS, std::min(BaseState::threads_, num_parallel_ops));
            }
            if (th_num > 1) {
              // Enable nested LAPACK to improve SVD
              save = omp_get_max_active_levels();
              omp_set_max_active_levels(5);
            }
            #pragma omp parallel for if (th_num > 1) schedule(dynamic, 1) num_threads(th_num)
            for (auto it_p = first_p_op; it_p <= last_p_op; ++it_p) {
              apply_op(*it_p, result, rng, final_ops and (it_p == last));
            }

            if (th_num > 1) {  // Restore for next operations
              omp_set_max_active_levels(save);
            }

            it = last_p_op;
          } else {
            apply_op(*it, result, rng, final_ops && (it + 1 == last));
         }
        } else {
          apply_op(*it, result, rng, final_ops && (it + 1 == last));
        }
      }
    }
  }
};

//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_op(const Operations::Op &op, ExperimentResult &result,
                     RngEngine &rng, bool final_op) {
  if (BaseState::creg().check_conditional(op)) {
    switch (op.type) {
    case OpType::barrier:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      apply_reset(op.qubits, rng);
      break;
    case OpType::initialize:
      apply_initialize(op.qubits, op.params, rng);
      break;
    case OpType::measure:
      apply_measure(op.qubits, op.memory, op.registers, rng);
      break;
    case OpType::bfunc:
      BaseState::creg().apply_bfunc(op);
      break;
    case OpType::roerror:
      BaseState::creg().apply_roerror(op, rng);
      break;
    case OpType::gate:
      apply_gate(op);
      break;
    case OpType::matrix:
      apply_matrix(op.qubits, op.mats[0]);
      break;
    case OpType::diagonal_matrix:
      BaseState::qreg_.apply_diagonal_matrix(op.qubits, op.params);
      break;
    case OpType::kraus:
      apply_kraus(op.qubits, op.mats, rng);
      break;
    case OpType::set_statevec: {
      reg_t all_qubits(qreg_.num_qubits());
      std::iota(all_qubits.begin(), all_qubits.end(), 0);
      qreg_.apply_initialize(all_qubits, op.params, rng);
      break;
    }
    case OpType::set_mps:
      qreg_.initialize_from_mps(op.mps);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      BaseState::apply_save_expval(op, result);
      break;
    case OpType::save_densmat:
      apply_save_density_matrix(op, result);
      break;
    case OpType::save_statevec:
      apply_save_statevector(op, result);
      break;
    case OpType::save_state:
    case OpType::save_mps:
      apply_save_mps(op, result, final_op);
      break;
    case OpType::save_probs:
    case OpType::save_probs_ket:
      apply_save_probs(op, result);
      break;
    case OpType::save_amps:
    case OpType::save_amps_sq:
      apply_save_amplitudes(op, result);
      break;
    default:
      throw std::invalid_argument(
          "MatrixProductState::State::invalid instruction \'" + op.name +
          "\'.");
    }
    // qreg_.print(std::cout);
    //  print out bond dimensions only if they may have changed since previous
    //  print
    if (MPS::get_mps_log_data() &&
        (op.type == OpType::gate || op.type == OpType::measure ||
         op.type == OpType::initialize || op.type == OpType::reset ||
         op.type == OpType::matrix) &&
        op.qubits.size() > 1) {
      output_bond_dimensions(op);
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

void State::apply_save_mps(const Operations::Op &op, ExperimentResult &result,
                           bool last_op) {
  if (op.qubits.size() != qreg_.num_qubits()) {
    throw std::invalid_argument(
        "Save MPS was not applied to all qubits."
        " Only the full matrix product state can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_") ? "matrix_product_state"
                                                        : op.string_params[0];
  if (last_op) {
    result.save_data_pershot(creg(), key, qreg_.move_to_mps_container(),
                             OpType::save_mps, op.save_type);
  } else {
    result.save_data_pershot(creg(), key, qreg_.copy_to_mps_container(),
                             OpType::save_mps, op.save_type);
  }
}

void State::apply_save_probs(const Operations::Op &op,
                             ExperimentResult &result) {
  rvector_t probs;
  qreg_.get_probabilities_vector(probs, op.qubits);
  if (op.type == OpType::save_probs_ket) {
    result.save_data_average(
        creg(), op.string_params[0],
        Utils::vec2ket(probs, MPS::get_json_chop_threshold(), 16), op.type,
        op.save_type);
  } else {
    result.save_data_average(creg(), op.string_params[0], std::move(probs),
                             op.type, op.save_type);
  }
}

void State::apply_save_amplitudes(const Operations::Op &op,
                                  ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save amplitudes instructions (empty params).");
  }
  Vector<complex_t> amps = qreg_.get_amplitude_vector(op.int_params);
  if (op.type == OpType::save_amps_sq) {
    // Square amplitudes
    std::vector<double> amps_sq(op.int_params.size());
    std::transform(amps.data(), amps.data() + amps.size(), amps_sq.begin(),
                   [](complex_t val) -> double { return pow(abs(val), 2); });
    result.save_data_average(creg(), op.string_params[0], std::move(amps_sq),
                             op.type, op.save_type);
  } else {
    result.save_data_pershot(creg(), op.string_params[0], std::move(amps),
                             op.type, op.save_type);
  }
}

double State::expval_pauli(const reg_t &qubits, const std::string &pauli) {
  return BaseState::qreg_.expectation_value_pauli(qubits, pauli).real();
}

void State::apply_save_statevector(const Operations::Op &op,
                                   ExperimentResult &result) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(
        "Save statevector was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  result.save_data_pershot(creg(), op.string_params[0],
                           qreg_.full_statevector(), op.type, op.save_type);
}

void State::apply_save_density_matrix(const Operations::Op &op,
                                      ExperimentResult &result) {
  cmatrix_t reduced_state;
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    reduced_state[0] = qreg_.norm();
  } else {
    reduced_state = qreg_.density_matrix(op.qubits);
  }

  result.save_data_average(creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

void State::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name +
        "\'.");
  switch (it->second) {
  case Gates::ccx:
    qreg_.apply_ccx(op.qubits);
    break;
  case Gates::cswap:
    qreg_.apply_cswap(op.qubits);
    break;
  case Gates::u3:
    qreg_.apply_u3(op.qubits[0], std::real(op.params[0]),
                   std::real(op.params[1]), std::real(op.params[2]));
    break;
  case Gates::u2:
    qreg_.apply_u2(op.qubits[0], std::real(op.params[0]),
                   std::real(op.params[1]));
    break;
  case Gates::u1:
    qreg_.apply_u1(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::cx:
    qreg_.apply_cnot(op.qubits[0], op.qubits[1]);
    break;
  case Gates::id: {
    break;
  }
  case Gates::x:
    qreg_.apply_x(op.qubits[0]);
    break;
  case Gates::y:
    qreg_.apply_y(op.qubits[0]);
    break;
  case Gates::z:
    qreg_.apply_z(op.qubits[0]);
    break;
  case Gates::h:
    qreg_.apply_h(op.qubits[0]);
    break;
  case Gates::s:
    qreg_.apply_s(op.qubits[0]);
    break;
  case Gates::sdg:
    qreg_.apply_sdg(op.qubits[0]);
    break;
  case Gates::sx:
    qreg_.apply_sx(op.qubits[0]);
    break;
  case Gates::sxdg:
    qreg_.apply_sxdg(op.qubits[0]);
    break;
  case Gates::t:
    qreg_.apply_t(op.qubits[0]);
    break;
  case Gates::tdg:
    qreg_.apply_tdg(op.qubits[0]);
    break;
  case Gates::r:
    qreg_.apply_r(op.qubits[0], std::real(op.params[0]),
                  std::real(op.params[1]));
    break;
  case Gates::rx:
    qreg_.apply_rx(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::ry:
    qreg_.apply_ry(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::rz:
    qreg_.apply_rz(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::swap:
    qreg_.apply_swap(op.qubits[0], op.qubits[1], true);
    break;
  case Gates::cy:
    qreg_.apply_cy(op.qubits[0], op.qubits[1]);
    break;
  case Gates::cz:
    qreg_.apply_cz(op.qubits[0], op.qubits[1]);
    break;
  case Gates::csx:
    qreg_.apply_csx(op.qubits[0], op.qubits[1]);
    break;
  case Gates::cu1:
    qreg_.apply_cu1(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::rxx:
    qreg_.apply_rxx(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::ryy:
    qreg_.apply_ryy(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::rzz:
    qreg_.apply_rzz(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::rzx:
    qreg_.apply_rzx(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::pauli:
    apply_pauli(op.qubits, op.string_params[0]);
    break;
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name +
        "\'.");
  }
}

void State::apply_pauli(const reg_t &qubits, const std::string &pauli) {
  const auto size = qubits.size();
  for (size_t i = 0; i < qubits.size(); ++i) {
    const auto qubit = qubits[size - 1 - i];
    switch (pauli[i]) {
    case 'I':
      break;
    case 'X':
      BaseState::qreg_.apply_x(qubit);
      break;
    case 'Y':
      BaseState::qreg_.apply_y(qubit);
      break;
    case 'Z':
      BaseState::qreg_.apply_z(qubit);
      break;
    default:
      throw std::invalid_argument("invalid Pauli \'" +
                                  std::to_string(pauli[i]) + "\'.");
    }
  }
}

void State::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (!qubits.empty() && mat.size() > 0)
    qreg_.apply_matrix(qubits, mat);
}

void State::apply_matrix(const reg_t &qubits, const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    qreg_.apply_diagonal_matrix(qubits, vmat);
  } else {
    qreg_.apply_matrix(qubits, vmat);
  }
}

void State::apply_kraus(const reg_t &qubits,
                        const std::vector<cmatrix_t> &kmats, RngEngine &rng) {
  qreg_.apply_kraus(qubits, kmats, rng);
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_initialize(const reg_t &qubits, const cvector_t &params,
                             RngEngine &rng) {
  qreg_.apply_initialize(qubits, params, rng);
}

void State::apply_measure(const reg_t &qubits, const reg_t &cmemory,
                          const reg_t &cregister, RngEngine &rng) {
  rvector_t rands;
  rands.reserve(qubits.size());
  for (int_t i = 0; i < qubits.size(); ++i)
    rands.push_back(rng.rand(0., 1.));
  reg_t outcome = qreg_.apply_measure(qubits, rands);
  creg().store_measure(outcome, cmemory, cregister);
}

rvector_t State::measure_probs(const reg_t &qubits) const {
  rvector_t probvector;
  qreg_.get_probabilities_vector(probvector, qubits);
  return probvector;
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits, uint_t shots,
                                         RngEngine &rng) {
  // There are two alternative algorithms for sample measure
  // We choose the one that is optimal relative to the total number
  // of qubits,and the number of shots.
  // The parameters used below are based on experimentation.
  // The user can override this by setting the parameter
  // "mps_sample_measure_algorithm"
  if (MPS::get_sample_measure_alg() == Sample_measure_alg::PROB &&
      qubits.size() == qreg_.num_qubits()) {
    return sample_measure_all(shots, rng);
  }
  return sample_measure_using_apply_measure(qubits, shots, rng);
}

std::vector<reg_t>
State::sample_measure_using_apply_measure(const reg_t &qubits, uint_t shots,
                                          RngEngine &rng) {
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);
  std::vector<rvector_t> rnds_list;
  rnds_list.reserve(shots);
  for (int_t i = 0; i < shots; ++i) {
    rvector_t rands;
    rands.reserve(qubits.size());
    for (int_t j = 0; j < qubits.size(); ++j)
      rands.push_back(rng.rand(0., 1.));
    rnds_list.push_back(rands);
  }

#pragma omp parallel if (BaseState::threads_ > 1)                              \
    num_threads(BaseState::threads_)
  {
    MPS temp;
#pragma omp for
    for (int_t i = 0; i < static_cast<int_t>(shots); i++) {
      temp.initialize(qreg_);
      auto single_result = temp.apply_measure_internal(qubits, rnds_list[i]);
      all_samples[i] = single_result;
    }
  }
  return all_samples;
}

std::vector<reg_t> State::sample_measure_all(uint_t shots, RngEngine &rng) {
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);

  for (uint_t i = 0; i < shots; i++) {
    auto single_result = qreg_.sample_measure(shots, rng);
    all_samples[i] = single_result;
  }
  return all_samples;
}

void State::apply_reset(const reg_t &qubits, RngEngine &rng) {
  qreg_.reset(qubits, rng);
}

std::pair<uint_t, double> State::sample_measure_with_prob(const reg_t &qubits,
                                                          RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);

  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

//-------------------------------------------------------------------------
} // end namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
