
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
                if value_info.name == input_name:
                    dims = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    tree[node.name]["inputs"].append({"name": input_name, "dims": dims})
                    break

        # Collect output dimensions
        for output_name in node.output:
            for value_info in model.graph.value_info:
                if value_info.name == output_name:
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
            # Count occurrences of each operator type with dimensions
            if op_type not in operator_counts:
                operator_counts[op_type] = {"dimensions": []}
            unique_node = {
                            "inputs": [i["dims"] for i in details["inputs"]],
                            "outputs": [o["dims"] for o in details["outputs"]],
                            "weights": [w["dims"] for w in details["weights"]]
                        }
            found_node = False
            for idx, dim in enumerate(operator_counts[op_type]["dimensions"]):
                if unique_node == {dim_key: dim_value for dim_key, dim_value in dim.items() if dim_key in unique_node}:
                    operator_counts[op_type]["dimensions"][idx]["count"] += 1
                    operator_counts[op_type]["dimensions"][idx]["node_names"].append({"node": {"name": node_name, "inputs": [i["name"] for i in details["inputs"]], "outputs": [o["name"] for o in details["outputs"]], "weights": [w["name"] for w in details["weights"]]}})
                    found_node = True
                    break
            if not found_node:
                operator_counts[op_type]["dimensions"].append(unique_node)
                operator_counts[op_type]["dimensions"][-1]["count"] = 1
                operator_counts[op_type]["dimensions"][-1]["node_names"] = [{"node": {"name": node_name, "inputs": [i["name"] for i in details["inputs"]], "outputs": [o["name"] for o in details["outputs"]], "weights": [w["name"] for w in details["weights"]]}}]

    # Save the operator counts to a JSON file
    operator_counts_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}_operator_counts.json")
    with open(operator_counts_file_path, "w") as operator_counts_file:
        json.dump(operator_counts, operator_counts_file, indent=4)

def process_and_split_operators(onnx_file, output_dir):
    operator_counts_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}_operator_counts.json")
    if not os.path.exists(operator_counts_file_path):
        raise FileNotFoundError(f"Operator counts file not found: {operator_counts_file_path}")

    with open(operator_counts_file_path, "r") as operator_counts_file:
        operator_counts = json.load(operator_counts_file)

    split_results = {}

    for op_type, details in operator_counts.items():
        if op_type in ["Add", "Mul", "Gemm", "MatMul", "Conv"]: # TODO: Add Convs
            for dimension_info in details["dimensions"]:
                inputs = dimension_info.get("inputs", [])
                outputs = dimension_info.get("outputs", [])
                weights = dimension_info.get("weights", [])

                if not inputs or not outputs or not weights:
                    # Skip if any of these are empty
                    continue
                if op_type == "Gemm":
                    # Extract M, K, and bias dimensions
                    batch = inputs[0][0] 
                    M = outputs[0][1]
                    K = inputs[0][1] 
                    bias = weights[1][0] 

                    if M != bias:
                        print("Weight rows don't match bias size")
                        print('Inputs:', inputs)
                        print('Outputs:', outputs)
                        print('Weights:', weights)
                    # Split MatMul into multiple matrix-vector multiplications along N
                    if "Gemv" not in split_results:
                        split_results["Gemv"] = [{
                            "M": M, # weight rows
                            "K": K, # weight columns
                            "batch": batch,
                            "reuse_type": "vector reuse",
                            "reuse_amount": M * batch,
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        }]
                        continue
                    found_result = False
                    for result in split_results["Gemv"]:
                        if result["M"] == M and result["K"] == K and result["batch"] == batch:
                            result["invokes"] += [node_name["node"]["name"] for node_name in dimension_info["node_names"]]
                            result["total_invokes"] += dimension_info["count"]
                            found_result = True
                            break
                    if not found_result:
                        split_results["Gemv"].append({
                        "M": M, # weight rows
                        "K": K, # weight columns
                        "batch": batch,
                        "reuse_type": "vector reuse",
                        "reuse_amount": M * batch,
                        "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                        "total_invokes": dimension_info["count"]
                        })
                elif op_type == "MatMul":
                    # Extract M and K dimensions
                    batch = inputs[0][0] 
                    M = outputs[0][1]
                    K = inputs[0][1] 

                    # Split MatMul into multiple matrix-vector multiplications along N
                    if "Gemv" not in split_results:
                        split_results["Gemv"] = [{
                            "M": M, # weight rows
                            "K": K, # weight columns
                            "batch": batch,
                            "reuse_type": "vector reuse",
                            "reuse_amount": M * batch,
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        }]
                        continue
                    found_result = False
                    for result in split_results["Gemv"]:
                        if result["M"] == M and result["K"] == K and result["batch"] == batch:
                            result["invokes"] += [node_name["node"]["name"] for node_name in dimension_info["node_names"]]
                            result["total_invokes"] += dimension_info["count"]
                            found_result = True
                            break
                    if not found_result:
                        split_results["Gemv"].append({
                        "M": M, # weight rows
                        "K": K, # weight columns
                        "batch": batch,
                        "reuse_type": "vector reuse",
                        "reuse_amount": M * batch,
                        "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                        "total_invokes": dimension_info["count"]
                        })
                elif op_type == "Add":
                    # Handle element-wise Add
                    M = weights[0][0] 
                    batch = inputs[0][0] if len(inputs[0]) > 0 else None
                    if "Eltwise-Add" not in split_results:
                        split_results["Eltwise-Add"] = [{
                            "M": M,
                            "batch": batch,
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        }]
                        continue
                    found_result = False
                    for result in split_results["Eltwise-Add"]:
                        if result["M"] == M and result["batch"] == batch:
                            result["invokes"] += [node_name["node"]["name"] for node_name in dimension_info["node_names"]]
                            result["total_invokes"] += dimension_info["count"]
                            found_result = True
                            break
                    if not found_result:
                        split_results["Eltwise-Add"].append({
                            "M": M,
                            "batch": batch,
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        })
                
                elif op_type == "Mul":
                    # Handle element-wise Mul
                    M = weights[0][0] 
                    batch = inputs[0][0] if len(inputs[0]) > 0 else None
                    if "Eltwise-Mul" not in split_results:
                        split_results["Eltwise-Mul"] = [{
                            "M": M,
                            "batch": batch,
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        }]
                        continue
                    found_result = False
                    for result in split_results["Eltwise-Mul"]:
                        if result["M"] == M and result["batch"] == batch:
                            result["invokes"] += [node_name["node"]["name"] for node_name in dimension_info["node_names"]]
                            result["total_invokes"] += dimension_info["count"]
                            found_result = True
                            break
                    if not found_result:
                        split_results["Eltwise-Mul"].append({
                            "M": M,
                            "batch": batch,
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        })
                elif op_type == "Conv":
                    # Extract the batch, image height, image width, and kernel dimensions
                    batch = inputs[0][0] 
                    channels = inputs[0][1]
                    image_height = inputs[0][2]
                    image_width = inputs[0][3]
                    num_filters = weights[0][0]
                    kernel_channels = weights[0][1]
                    kernel_height = weights[0][2]
                    kernel_width = weights[0][3]
                    
                    if channels != kernel_channels:
                        print("Input channels do not match kernel channels")
                        print('Inputs:', inputs)
                        print('Outputs:', outputs)
                        print('Weights:', weights)
                        continue

                    # A convolution can be converted into GEMM with dimensions
                    # PQ x CRS for the input activations, and CRS x K for the weights
                    # P/Q are the output image dims, C is channels, R/S are the kernel dims,
                    # and K is the number of filters

                    M = (image_height - kernel_height) * (image_width - kernel_width)
                    K = kernel_height * kernel_width * channels
                    reuse_amt = num_filters * batch * M
                    # Split MatMul into multiple matrix-vector multiplications along N
                    if "Gemv" not in split_results:
                        split_results["Gemv"] = [{
                            "M": M, # input activation rows
                            "K": K, # input activation columns
                            "batch": batch,
                            "reuse_type": "vector reuse",
                            "reuse_amount": reuse_amt, 
                            "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                            "total_invokes": dimension_info["count"]
                        }]
                        continue
                    found_result = False
                    for result in split_results["Gemv"]:
                        if result["M"] == M and result["K"] == K and result["batch"] == batch:
                            result["invokes"] += [node_name["node"]["name"] for node_name in dimension_info["node_names"]]
                            result["total_invokes"] += dimension_info["count"]
                            found_result = True
                            break
                    if not found_result:
                        split_results["Gemv"].append({
                        "M": M, # weight rows
                        "K": K, # weight columns
                        "batch": batch,
                        "reuse_type": "vector reuse",
                        "reuse_amount": reuse_amt,
                        "invokes": [node_name["node"]["name"] for node_name in dimension_info["node_names"]],
                        "total_invokes": dimension_info["count"]
                        })
                    
    # Save the split results to a JSON file
    split_results_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}_split_results.json")
    with open(split_results_file_path, "w") as split_results_file:
        json.dump(split_results, split_results_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Download ML model, convert to ONNX, and list operations.")
    parser.add_argument("model", choices=["ViT", "RNN-T", "LSTM", "GPT-2", "ResNet-50", "Star-GAN", "DORN", "All"],
                        help="Specify the ML model to process.")
    args = parser.parse_args()

    try:
        if args.model == "All":
            for model_name in ["ViT", "RNN-T", "LSTM", "GPT-2", "ResNet-50", "Star-GAN", "DORN"]:
                os.makedirs("onnx_models", exist_ok=True)
                operations_dir = "operations"
                os.makedirs(operations_dir, exist_ok=True)
                onnx_file = model_to_onnx(model_name)
                ops_tree = list_operations(onnx_file)
                # Save the tree to the operations directory
                json_file_path = os.path.join("operations", f"{os.path.splitext(os.path.basename(onnx_file))[0]}.json")
                with open(json_file_path, "w") as json_file:
                    json.dump(ops_tree, json_file, indent=4)
                # Filter and save operations with dimensions
                save_operations_with_dimensions(onnx_file, ops_tree, "operations")
                process_and_split_operators(onnx_file, "operations")
        else:
            # Ensure the directory exists
            os.makedirs("onnx_models", exist_ok=True)
            # Create a directory for operations JSON files
            operations_dir = "operations"
            os.makedirs(operations_dir, exist_ok=True)
            onnx_file = model_to_onnx(args.model)
            ops_tree = list_operations(onnx_file)
            # print("Listing operations in the ONNX model...")
            # for op in operations:
            #     print(op)
            # Save the tree to the operations directory
            json_file_path = os.path.join(operations_dir, f"{os.path.splitext(os.path.basename(onnx_file))[0]}.json")
            with open(json_file_path, "w") as json_file:
                json.dump(ops_tree, json_file, indent=4)
            # Filter and save operations with dimensions
            save_operations_with_dimensions(onnx_file, ops_tree, operations_dir)
            process_and_split_operators(onnx_file, operations_dir)
    except NotImplementedError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()