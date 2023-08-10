# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""

from ddt import ddt
from test.terra.reference import ref_measure
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, depolarizing_error
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info.random import random_unitary
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

SUPPORTED_METHODS = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestMeasure(SimulatorTestCase):
    """AerSimulator measure tests."""

    OPTIONS = {"seed_simulator": 41411}

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_with_sampling(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        target_counts = ref_measure.measure_counts_deterministic(shots)
        target_memory = ref_measure.measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_without_sampling(self, method, device):
        """Test AerSimulator measure with deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=False)
        target_counts = ref_measure.measure_counts_deterministic(shots)
        target_memory = ref_measure.measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_with_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_without_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        delta = 0.05
        if "tensor_network" in method:
            shots = 100
            delta = 0.1
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_sampling_with_readouterror(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_sampling_with_quantum_noise(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        depolarizing = {"u3": (1, 0.001), "cx": (2, 0.02)}
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in depolarizing.items():
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(gate_error, num_qubits), gate
            )

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        sampling = method == "density_matrix" or method == "tensor_network"
        self.compare_result_metadata(result, circuits, "measure_sampling", sampling)

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_multi_qubit_with_sampling(self, method, device):
        """Test AerSimulator multi-qubit measure with deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(allow_sampling=True)
        target_counts = ref_measure.multiqubit_measure_counts_deterministic(shots)
        target_memory = ref_measure.multiqubit_measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_multi_qubit_without_sampling(self, method, device):
        """Test AerSimulator multi-qubit measure with deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(allow_sampling=False)
        target_counts = ref_measure.multiqubit_measure_counts_deterministic(shots)
        target_memory = ref_measure.multiqubit_measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_multi_qubit_with_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_multi_qubit_without_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        delta = 0.05
        if "tensor_network" in method:
            shots = 100
            delta = 0.1
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    # ---------------------------------------------------------------------
    # Test MPS algorithms for measure
    # ---------------------------------------------------------------------
    def test_mps_measure_alg_qv(self):
        """Test MPS measure algorithms with quantum volume"""
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 5
        depth = 2
        circuit = QuantumVolume(n, depth, seed=9)
        circuit.measure_all()
        circuit = transpile(circuit, backend)

        result1 = backend.run(
            circuit, shots=shots, mps_sample_measure_algorithm="mps_apply_measure"
        ).result()
        self.assertTrue(getattr(result1, "success", "True"))

        result2 = backend.run(
            circuit, shots=shots, mps_sample_measure_algorithm="mps_probabilities"
        ).result()
        self.assertTrue(getattr(result2, "success", "True"))

        self.assertDictAlmostEqual(
            result1.get_counts(circuit), result2.get_counts(circuit), delta=0.1 * shots
        )

    def test_mps_measure_subset_alg_qv(self):
        """Test MPS measure algorithms with quantum volume"""
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 5
        circuits = []
        for i in range(2):
            circuit = QuantumCircuit(n, n)
            circuit.unitary(random_unitary(4), [0, 1])
            circuit.unitary(random_unitary(4), [1, 2])
            circuit.unitary(random_unitary(4), [2, 3])
            circuit.unitary(random_unitary(4), [3, 4])
            circuits.append(circuit)
        circuits[0].measure([0, 2, 4], [0, 2, 4])
        circuits[1].measure([4, 1], [4, 1])
        circuits = transpile(circuits, backend)

        for circuit in circuits:
            result1 = backend.run(
                circuit, shots=shots, mps_sample_measure_algorithm="mps_apply_measure"
            ).result()
            self.assertTrue(getattr(result1, "success", "True"))

            result2 = backend.run(
                circuit, shots=shots, mps_sample_measure_algorithm="mps_probabilities"
            ).result()
            self.assertTrue(getattr(result2, "success", "True"))

            self.assertDictAlmostEqual(
                result1.get_counts(circuit), result2.get_counts(circuit), delta=0.1 * shots
            )

    def test_mps_measure_with_limited_bond_dimension(self):
        """Test MPS measure with limited bond dimension,
        where the qubits are not in sorted order
        """
        backend_statevector = self.backend(method="statevector")
        shots = 1000
        n = 4
        for bd in [2, 4]:
            backend_mps = self.backend(
                method="matrix_product_state", matrix_product_state_max_bond_dimension=bd
            )
            for measured_qubits in [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [2, 0, 1, 3],
                [0, 1, 2],
                [2, 1, 3],
                [1, 3, 0],
                [0, 2, 3],
            ]:
                circuit = QuantumCircuit(n, n)
                circuit.h(3)
                circuit.h(1)
                circuit.cx(1, 2)
                circuit.cx(3, 0)
                circuit.measure(measured_qubits, measured_qubits)
                res_mps = backend_mps.run(circuit, shots=shots).result().get_counts()
                self.assertTrue(getattr(res_mps, "success", "True"))
                res_sv = backend_statevector.run(circuit, shots=shots).result().get_counts()
                self.assertDictAlmostEqual(res_mps, res_sv, delta=0.1 * shots)

    def test_mps_parallel_ops_with_gate_circuits(self):
        """Test MPS parallel ops works properly"""
        from qiskit.circuit.library import QFT  # bad practice but for clarity

        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 30

        qc1 = QuantumCircuit(n)
        qiskit_qft = QFT(n).to_gate()
        qc1.append(qiskit_qft, range(n))
        qc1.measure_all()
        t_qc1 = transpile(qc1, backend, seed_transpiler=23)

        qc2 = QuantumCircuit(n, metadata={"skip_key_order": True, "parallel_ops": True})
        qiskit_qft = QFT(n).to_gate()
        qc2.append(qiskit_qft, range(n))
        qc2.measure_all()
        t_qc2 = transpile(qc2, backend, seed_transpiler=23)

        result1 = backend.run(t_qc1, shots=shots).result()
        self.assertTrue(getattr(result1, 'success', 'True'))

        result2 = backend.run(t_qc2, shots=shots).result()
        self.assertTrue(getattr(result2, 'success', 'True'))

        # Result should be the same because they should be the same exact circuit
        self.assertDictEqual(result1.get_counts(t_qc1), result2.get_counts(t_qc2))

    def test_skip_lexicographic_order_results(self):
        """ Check that the result is the same when skipping lexicographic order"""
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 12

        for checks in range(10):  # <-- Never trust parallel code
            qc1 = QuantumCircuit(n)
            qc2 = QuantumCircuit(n, metadata={"skip_key_order": True})

            for times in range(4):
                if times % 2 == 0:
                    for i in range(0, n, 2):
                        qc1.unitary(random_unitary(4, seed=23), [i, i+1])
                        qc2.unitary(random_unitary(4, seed=23), [i, i+1])
                else:
                    for i in range(1,n-1,2):
                        qc1.unitary(random_unitary(4, seed=11), [i, i+1])
                        qc2.unitary(random_unitary(4, seed=11), [i, i+1])

            qc1.measure_all()
            qc2.measure_all()

            qc1 = transpile(qc1, backend, optimization_level=1)
            qc2 = transpile(qc2, backend, optimization_level=1)

            result1 = backend.run(qc1, shots=shots).result()
            result2 = backend.run(qc2, shots=shots).result()

            self.assertDictEqual(result1.get_counts(qc1), result2.get_counts(qc2))

    def test_mps_order_and_parallel(self):
        """ Check that the result is the same when
            skiping the lexicographical ordering
        """
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 12

        for checks in range(10):  # <-- Never trust parallel execution
            qc1 = QuantumCircuit(n)
            qc2 = QuantumCircuit(n, metadata={"parallel_ops": True})

            for times in range(4):
                if times % 2 == 0:
                    for i in range(0, n, 2):
                        qc1.unitary(random_unitary(4, seed=23), [i, i+1])
                        qc2.unitary(random_unitary(4, seed=23), [i, i+1])
                else:
                    for i in range(1,n-1,2):
                        qc1.unitary(random_unitary(4, seed=11), [i, i+1])
                        qc2.unitary(random_unitary(4, seed=11), [i, i+1])

            qc1.measure_all()
            qc2.measure_all()

            qc1 = transpile(qc1, backend, optimization_level=1)
            qc2 = transpile(qc2, backend, optimization_level=1)

            result1 = backend.run(qc1, shots=shots).result()
            result2 = backend.run(qc2, shots=shots).result()

            self.assertTrue("matrix_product_state_parallel_ops" not in result1._get_experiment().metadata)
            self.assertTrue("matrix_product_state_parallel_ops" in result2._get_experiment().metadata)

            self.assertDictEqual(result1.get_counts(qc1), result2.get_counts(qc2))

    def test_mps_noorder_and_parallel(self):
        """ Check that the result is the same when
            skiping the lexicographical ordering
        """
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 12

        for checks in range(10):  # <-- Never trust parallel execution
            qc1 = QuantumCircuit(n)
            qc2 = QuantumCircuit(n, metadata={"skip_key_order": True, "parallel_ops": True})

            for times in range(4):
                if times % 2 == 0:
                    for i in range(0, n, 2):
                        qc1.unitary(random_unitary(4, seed=23), [i, i+1])
                        qc2.unitary(random_unitary(4, seed=23), [i, i+1])
                else:
                    for i in range(1,n-1,2):
                        qc1.unitary(random_unitary(4, seed=11), [i, i+1])
                        qc2.unitary(random_unitary(4, seed=11), [i, i+1])

            qc1.measure_all()
            qc2.measure_all()

            qc1 = transpile(qc1, backend, optimization_level=1)
            qc2 = transpile(qc2, backend, optimization_level=1)

            result1 = backend.run(qc1, shots=shots).result()
            result2 = backend.run(qc2, shots=shots).result()

            self.assertTrue("matrix_product_state_parallel_ops" not in result1._get_experiment().metadata)
            self.assertTrue("matrix_product_state_parallel_ops" in result2._get_experiment().metadata)

            self.assertDictEqual(result1.get_counts(qc1), result2.get_counts(qc2))

