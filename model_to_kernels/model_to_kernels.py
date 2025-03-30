
import onnx
from onnx import shape_inference

from cnn_to_onnx import cnn_to_onnx
from lstm_to_onnx import lstm_to_onnx
from gpt_to_onnx import gpt_to_onnx
from vit_to_onnx import vit_to_onnx
from rnnt_to_onnx import rnnt_to_onnx
from gan_to_onnx import stargan_to_onnx
from dorn_to_onnx import dorn_to_onnx

import argparse
import json
import os

KERNELS_TO_CHECK = ["Gemm", "MatMul", "Add", "Conv"]

def model_to_onnx(model_name, directory="onnx_models"):
    if model_name == "ResNet-50":
        return cnn_to_onnx(directory=directory)
    elif model_name == "LSTM":
        return lstm_to_onnx(directory=directory)
    elif model_name == "GPT-2":
        return gpt_to_onnx(directory=directory)
    elif model_name == "ViT":
        return vit_to_onnx(directory=directory)
    elif model_name == "RNN-T":
        return rnnt_to_onnx(directory=directory)
    elif model_name == "Star-GAN":
        return stargan_to_onnx(directory=directory)
    elif model_name == "DORN":
        return dorn_to_onnx(directory=directory)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def list_operations(onnx_file):
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)
    tree = {}

    # Create a tree structure showing how nodes lead to other ones
    for node in model.graph.node:
        tree[node.name] = {
            "op_type": node.op_type,
            "inputs": [],
            "outputs": [],
            "weights": []
        }
        # Collect input dimensions
        for input_name in node.input:
            for value_info in model.graph.value_info:
                if value_info.name == input_name or input_name == "input":
                    dims = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    tree[node.name]["inputs"].append({"name": input_name, "dims": dims})
                    break

        # Collect output dimensions
        for output_name in node.output:
            for value_info in model.graph.value_info:
                if value_info.name == output_name or output_name == "output":
                    dims = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    tree[node.name]["outputs"].append({"name": output_name, "dims": dims})
                    break

        # Collect weight dimensions
        for initializer in model.graph.initializer:
            if initializer.name in node.input:
                dims = list(initializer.dims)
                tree[node.name]["weights"].append({"name": initializer.name, "dims": dims})

    return tree

def save_operations_with_dimensions(onnx_file, ops_tree, output_dir):
    operator_counts = {}

    for node_name, details in ops_tree.items():
        if details["inputs"] or details["outputs"] or details["weights"]:
            op_type = details["op_type"]
            # Collect occurrences of each operator type with dimensions
            if op_type not in operator_counts:
                operator_counts[op_type] = []
            unique_node = {
                "inputs": [i["dims"] for i in details["inputs"]],
                "outputs": [o["dims"] for o in details["outputs"]],
                "weights": [w["dims"] for w in details["weights"]],
                "node_info": {"name": node_name, "inputs": [i["name"] for i in details["inputs"]], "outputs": [o["name"] for o in details["outputs"]], "weights": [w["name"] for w in details["weights"]]}
            }
            operator_counts[op_type].append(unique_node)

    # Save the operator counts to a JSON file
    operator_counts_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}_operator_groupings.json")
    with open(operator_counts_file_path, "w") as operator_counts_file:
        json.dump(operator_counts, operator_counts_file, indent=4)
    return operator_counts_file_path

def find_shared_inputs(ops_tree, onnx_file, output_dir):
    output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}.json")
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)
    input_usage = {}
    input_distances = {}

    # Track input usage across nodes
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in input_usage:
                input_usage[input_name] = []
            input_usage[input_name].append(node)

    # Calculate distances of inputs from the model's input tensors
    for value_info in model.graph.input:
        input_name = value_info.name
        input_distances[input_name] = 0  # Inputs are at distance 0

    for node in model.graph.node:
        for input_name in node.input:
            if input_name in input_distances:
                distance = input_distances[input_name] + 1
                for output_name in node.output:
                    if output_name not in input_distances or distance > input_distances[output_name]:
                        input_distances[output_name] = distance

    shared_inputs = {}
    skipped_inputs = {}

    # Identify inputs shared across multiple GEMM, MatMul, Add, or Conv nodes
    for input_name, nodes in input_usage.items():
        relevant_nodes = [node for node in nodes if node.op_type in KERNELS_TO_CHECK]
        if len(relevant_nodes) > 1:
            # Check if all nodes are at the same distance from the input. The size of the set will be 1 if all nodes are at the same distance
            distances = {input_distances.get(node.input[0], -1) for node in relevant_nodes if node.input} 
            # Check that all of the sizes of the outputs are the same. The size of the set will be 1 if all nodes have the same output size
            output_sizes = {tuple(o["dims"]) for node in relevant_nodes for o in ops_tree[node.name]["outputs"]}
            # Check if all relevant nodes have the same op_type
            op_types = {node.op_type for node in relevant_nodes}
            if len(distances) == 1 and '/' in input_name and len(output_sizes) == 1 and len(op_types) == 1:
                shared_inputs[input_name] = {
                    "nodes": [{"name": node.name, "op_type": node.op_type} for node in relevant_nodes],
                }
            else:
                skipped_inputs[input_name] = {
                    "nodes": [{"name": node.name, "op_type": node.op_type} for node in relevant_nodes],
                }

    # Save shared and skipped inputs to a JSON file
    shared_and_skipped_inputs = {
        "shared_inputs": shared_inputs,
        "skipped_inputs": skipped_inputs
    }
    with open(output_file_path, "w") as output_file:
        json.dump(shared_and_skipped_inputs, output_file, indent=4)
    return output_file_path

def find_reuse(model_name, onnx_file, shared_inputs_file_path, operator_counts_file_path, output_dir, kernels_to_check=KERNELS_TO_CHECK):
    if not os.path.exists(operator_counts_file_path):
        raise FileNotFoundError(f"Operator counts file not found: {operator_counts_file_path}")
    if not os.path.exists(shared_inputs_file_path):
        raise FileNotFoundError(f"Shared inputs file not found: {shared_inputs_file_path}")

    with open(operator_counts_file_path, "r") as operator_counts_file:
        operator_counts = json.load(operator_counts_file)
        # Go through the shared inputs
        with open(shared_inputs_file_path, "r") as shared_inputs_file:
            shared_inputs_data = json.load(shared_inputs_file)
            shared_inputs = shared_inputs_data.get("shared_inputs", {})
            reuse_results = {}
            skipped_iterations = []
            for input_name, details in shared_inputs.items():
                print("Processing shared input:", input_name)
                nodes = details.get("nodes", [])
                if nodes[0]["op_type"] not in kernels_to_check:
                    print(f"Skipping input {input_name} as it is not in the kernels to check: {kernels_to_check}")
                    continue
                dimensions = {"inputs": [], "outputs": [], "weights": []}

                if len(nodes) > 1 and not all(node["op_type"] == nodes[0]["op_type"] for node in nodes):
                    raise ValueError(f"Currently expecting all nodes sharing inputs to have the same op_type. Shared input: {input_name}, nodes: {nodes}")

                for node in nodes:
                    node_name = node.get("name")
                    op_type = node.get("op_type")

                    if op_type in operator_counts:
                        for item in operator_counts[op_type]:
                            if item["node_info"]["name"] == node_name:
                                dimensions["inputs"].append(item.get("inputs", []))
                                dimensions["outputs"].append(item.get("outputs", []))
                                dimensions["weights"].append(item.get("weights", []))
                                break

                # Check if all dimensions match. For now we expect them to be the same across all nodes that share the same input
                dimensions_match = True
                for dim_list in dimensions.values():
                    if len(dim_list) > 1 and not all(dim == dim_list[0] for dim in dim_list):
                        skipped_iterations.append(input_name)
                        dimensions_match = False
                        break
                        # raise ValueError(f"Currenly expecting dimensions to match for shared inputs' nodes. Shared input: {input_name}, nodes: {nodes}")
                if not dimensions_match:
                    print(f"Dimensions do not match for shared input: {input_name}. Skipping...")
                    continue

                # Only need to check one node that's using the shared input since we expect the rest that share the same input
                # should have the same dimensions
                inputs_0 = dimensions["inputs"][0] if dimensions["inputs"] else []
                if not dimensions["weights"] and not dimensions["inputs"][0]:
                    print("No weights or first input found for shared input:", input_name)
                    continue
                if not dimensions["weights"] and not len(dimensions["inputs"]) > 1:
                    print("No weights or second input found for shared input:", input_name)
                    continue
                if not dimensions["inputs"][0] and not dimensions["outputs"][0]:
                    print("No inputs or outputs found for shared input:", input_name)
                    continue
                if dimensions["weights"]:
                    inputs_1 = dimensions["weights"][0]
                else: 
                    if dimensions["inputs"][1]: 
                        inputs_1 = dimensions["inputs"][1]
                    else:
                        inputs_1 = [[dimensions["outputs"][-1], dimensions["outputs"][-2]]] # Guess from the output
                outputs = dimensions["outputs"][0] if dimensions["outputs"] else []
                op_type = nodes[0]["op_type"]
                if op_type == "Gemm":
                    batch = inputs_0[0][0]
                    input_size = inputs_0[0][1]
                    output_size = inputs_1[0][0]
                    # If batching, then the reused input is a matrix, otherwise it's a vector
                    reuse_type = "matrix" if batch > 1 else "vector"
                    # For Gemm, the reuse amount is the number of nodes that share the input matrix times the output size
                    # since the input matrix is being reused
                    reuse_amount = len(nodes) * output_size

                    if "Gemv" not in reuse_results:
                        reuse_results["Gemv"] = []
                    reuse_results["Gemv"].append({
                        "M": batch,
                        "K": input_size,
                        "reuse_type": reuse_type,
                        "reuse_amount": reuse_amount,
                    })
                elif op_type == "MatMul":
                    batch = inputs_0[0][0]
                    input_size = inputs_0[0][1]
                    output_size = inputs_1[0][0]
                    # If batching, then the reused input is a matrix, otherwise it's a vector
                    reuse_type = "matrix" if batch > 1 else "vector"
                    # For Gemm, the reuse amount is the number of nodes that share the input matrix times the output size
                    # since the input matrix is being reused
                    reuse_amount = len(nodes) * output_size

                    if "Gemv" not in reuse_results:
                        reuse_results["Gemv"] = []
                    reuse_results["Gemv"].append({
                        "M": batch,
                        "K": input_size,
                        "reuse_type": reuse_type,
                        "reuse_amount": reuse_amount,
                    })
                elif op_type == "Add":
                    batch = inputs_0[0][0]
                    input_size = inputs_0[0][1]
                    reuse_type = "matrix" if batch > 1 else "vector"
                    # For element-wise add, the reuse amount is the number of nodes that share the input matrix
                    # The batch size and vector size of the input will have the same dimensions as the 
                    # weight and output matrices
                    reuse_amount = len(nodes)

                    if "EltwiseAdd" not in reuse_results:
                        reuse_results["EltwiseAdd"] = []
                    reuse_results["EltwiseAdd"].append({
                        "M": batch,
                        "K": input_size,
                        "reuse_type": reuse_type,
                        "reuse_amount": reuse_amount,
                    })
                elif op_type == "Conv":
                    batch = outputs[0][0]
                    output_height = outputs[0][2]
                    output_width = outputs[0][3]
                    output_channels = weights[0][0]
                    input_channels = weights[0][1]
                    kernel_height = weights[0][2]
                    kernel_width = weights[0][3]
                    # The M, K here are the matrix dimensions of the input activation matrix 
                    M = batch * output_height * output_width
                    K = kernel_height * kernel_width * input_channels
                    reuse_type = "matrix"

                    # Since the input activation matrix is being reused, the filters are split into vectors, and
                    # so part of the reuse amount is the number of filters
                    reuse_amount = output_channels

                    if "Gemv" not in reuse_results:
                        reuse_results["Gemv"] = []
                    reuse_results["Gemv"].append({
                        "M": M,
                        "K": K,
                        "reuse_type": "matrix",
                        "reuse_amount": reuse_amount,
                    })
            # Collect all node names in shared inputs. This will be used to find nodes that don't have shared input
            # but still can be considered for other reuse opportunities
            all_node_names = []
            for input_name, details in shared_inputs.items():
                nodes = details.get("nodes", [])
                for node in nodes:
                    node_name = node.get("name")
                    if node_name:
                        all_node_names.append(node_name)
            all_node_names = set(all_node_names) - set(skipped_iterations)
            # Process nodes that are not part of shared inputs but are in operator_counts
            for op_type, nodes in operator_counts.items():
                if op_type not in kernels_to_check:
                    print(f"Skipping operator {op_type} as it is not in the kernels to check: {kernels_to_check}")
                    continue

                for node in nodes:
                    node_name = node["node_info"]["name"]
                    if node_name in all_node_names:
                        continue  # Skip nodes already processed in shared inputs
                    print("Processing node:", node_name)

                    inputs = node.get("inputs", [])
                    outputs = node.get("outputs", [])
                    weights = node.get("weights", [])
                    if not weights and not inputs[0]:
                        print("No weights or first input found for:", node_name)
                        continue
                    if not weights and not len(inputs) > 1:
                        print("No weights or second input found for:", node_name)
                        continue
                    if not inputs[0] and not outputs[0]:
                        print("No inputs or outputs found for:", node_name)
                        continue

                    if op_type == "Gemm":
                        batch = inputs[0][0]
                        input_size = inputs[0][-2]
                        if weights:
                            output_size = weights[0][0]
                        else:
                            if inputs[1]:
                                output_size = inputs[1][-1]
                            else:
                                output_size = outputs[0][-1]
                        # If batching, then the reused input is a matrix, otherwise it's a vector
                        reuse_type = "matrix" if batch > 1 else "vector"
                        # For Gemm, the reuse amount is the output size since the input matrix is being reused
                        reuse_amount = output_size

                        if "Gemv" not in reuse_results:
                            reuse_results["Gemv"] = []
                        reuse_results["Gemv"].append({
                            "M": batch,
                            "K": input_size,
                            "reuse_type": reuse_type,
                            "reuse_amount": reuse_amount,
                        })
                    elif op_type == "MatMul":
                        batch = inputs[0][0]
                        input_size = inputs[0][-2]
                        if weights:
                            output_size = weights[0][0]
                        else:
                            if inputs[1]:
                                output_size = inputs[1][-1]
                            else:
                                output_size = outputs[0][-1]
                        # If batching, then the reused input is a matrix, otherwise it's a vector
                        reuse_type = "matrix" if batch > 1 else "vector"
                        # For Gemm, the reuse amount is the output size since the input matrix is being reused
                        reuse_amount = output_size

                        if "Gemv" not in reuse_results:
                            reuse_results["Gemv"] = []
                        reuse_results["Gemv"].append({
                            "M": batch,
                            "K": input_size,
                            "reuse_type": reuse_type,
                            "reuse_amount": reuse_amount,
                        })
                    elif op_type == "Add":
                        batch = inputs[0][0]
                        input_size = inputs[0][1]
                        reuse_type = "matrix" if batch > 1 else "vector"
                        # For element-wise add, there's no reuse amount
                        # The batch size and vector size of the input will have the same dimensions as the 
                        # weight and output matrices
                        reuse_amount = 1

                        if "EltwiseAdd" not in reuse_results:
                            reuse_results["EltwiseAdd"] = []
                        reuse_results["EltwiseAdd"].append({
                            "M": batch,
                            "K": input_size,
                            "reuse_type": reuse_type,
                            "reuse_amount": reuse_amount,
                        })
                    elif op_type == "Conv":
                        batch = outputs[0][0]
                        output_height = outputs[0][2]
                        output_width = outputs[0][3]
                        output_channels = weights[0][0]
                        input_channels = weights[0][1]
                        kernel_height = weights[0][2]
                        kernel_width = weights[0][3]
                        # The M, K here are the matrix dimensions of the input activation matrix 
                        M = batch * output_height * output_width
                        K = kernel_height * kernel_width * input_channels
                        reuse_type = "matrix"

                        # Since the input activation matrix is being reused, the filters are split into vectors, and
                        # so part of the reuse amount is the number of filters
                        reuse_amount = output_channels

                        if "Gemv" not in reuse_results:
                            reuse_results["Gemv"] = []
                        reuse_results["Gemv"].append({
                            "M": M,
                            "K": K,
                            "reuse_type": reuse_type,
                            "reuse_amount": reuse_amount,
                        })
    # Save the split results to a JSON file
    reuse_results_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}_reuse_results.json")
    with open(reuse_results_file_path, "w") as reuse_results_file:
        json.dump(reuse_results, reuse_results_file, indent=4)
    return reuse_results_file_path

def reuse_results_to_csv(onnx_file, reuse_results_file_path, operations_dir):
    csv_file_path = os.path.join(operations_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}.txt")
    with open(csv_file_path, "w") as csv_file:
        csv_file.write("Operator,M,N,Reuse Type,Reuse Amount\n")  # Write CSV header
        with open(reuse_results_file_path, "r") as json_file:
            split_results = json.load(json_file)
            for op_type, details in split_results.items():
                for item in details:
                    M = item.get("M", "")
                    K = item.get("K", "")
                    reuse_type = item.get("reuse_type", "")
                    reuse_amount = item.get("reuse_amount", "")
                    csv_file.write(f"{op_type},{M},{K},{reuse_type},{reuse_amount}\n")
    return csv_file_path

def main():
    parser = argparse.ArgumentParser(description="Download ML model, convert to ONNX, and list operations.")
    parser.add_argument("model", choices=["ViT", "LSTM", "RNN-T", "GPT-2", "ResNet-50", "Star-GAN", "DORN", "All"], 
                        help="Specify the ML model to process.")
    args = parser.parse_args()

    try:
        for model_name in ["ViT", "LSTM", "GPT-2", "ResNet-50", "Star-GAN", "DORN"]: # Removed RNN-T for now because errors pop up one after another
            if args.model == "All" or args.model == model_name:
                onnx_models_dir = "onnx_models"
                operations_dir = "operations"
                shared_inputs_dir = "shared_inputs"
                reuse_results_dir = "reuse_results"
                cost_model_input_dir = "cost_model_input"
                directories = [onnx_models_dir, operations_dir, shared_inputs_dir, reuse_results_dir, cost_model_input_dir]
                # Create directories if they don't exist
                for directory in directories:
                    os.makedirs(directory, exist_ok=True)
                onnx_file = model_to_onnx(model_name, onnx_models_dir)
                ops_tree = list_operations(onnx_file)
                # print("Listing operations in the ONNX model...")
                # for op in operations:
                #     print(op)
                # Save the tree to the operations directory
                json_file_path = os.path.join(operations_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}.json")
                with open(json_file_path, "w") as json_file:
                    json.dump(ops_tree, json_file, indent=4)
                    
                operator_counts_file_path = save_operations_with_dimensions(onnx_file, ops_tree, operations_dir)
                # Find nodes with outputs that are shared across multiple GEMM, MatMul, Add, or Conv nodes
                shared_inputs_file_path = find_shared_inputs(ops_tree, onnx_file, shared_inputs_dir)
                reuse_results_file_path = find_reuse(model_name, onnx_file, shared_inputs_file_path, operator_counts_file_path, reuse_results_dir)
                csv_file_path = reuse_results_to_csv(onnx_file, reuse_results_file_path, cost_model_input_dir)
                print(f"CSV file with reuse results for ONNX file {onnx_file} saved at: {csv_file_path}")
    except NotImplementedError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
