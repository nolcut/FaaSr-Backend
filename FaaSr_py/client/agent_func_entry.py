import logging
import sys
import traceback as tb_module
from pathlib import Path

from FaaSr_py.client.py_client_stubs import faasr_exit, faasr_log, faasr_return
from FaaSr_py.helpers.agent_constraints import AgentContextManager
from FaaSr_py.helpers.agent_helper import (
    AgentCodeGenerator,
    get_agent_api_key,
    get_agent_provider,
)

logger = logging.getLogger(__name__)


def run_agent_function(faasr, prompt, action_name):
    """
    Entry point for agent function execution with adaptive exploration

    Arguments:
        faasr: FaaSr payload instance
        prompt: Natural language prompt for the agent
        action_name: Name of the action being executed
    """
    logger.info(f"Starting agent execution for {action_name} with prompt: {prompt[:100]}")

    # Initialize constraints
    context = AgentContextManager(faasr)

    try:
        # Get API key and provider
        api_key = get_agent_api_key()
        provider = get_agent_provider()

        if not provider:
            raise RuntimeError(
                "Could not determine LLM provider. Please set AGENT_KEY with valid OpenAI or Claude key."
            )

        logger.info(f"Using {provider} as LLM provider")

        # First, explore the S3 bucket to provide context
        exploration_data = _explore_s3_context(faasr, context)
        
        # Generate code from prompt with exploration context
        generator = AgentCodeGenerator(api_key, provider)
        code = generator.generate_code_with_context(prompt, exploration_data)

        # Log generated code to S3
        _log_generated_code_to_s3(faasr, action_name, code)

        # Validate code safety
        if not generator.validate_code_safety(code):
            err_msg = "Generated code failed safety validation"
            logger.error(err_msg)
            faasr_exit(err_msg)

        logger.debug(f"Generated code passed safety checks")

        # Prepare execution environment with adaptive capability
        agent_namespace = _prepare_agent_namespace(faasr, context)
        
        # Add adaptive regeneration capability
        agent_namespace["_regenerate_approach"] = lambda new_prompt, discovered_data: _adaptive_regeneration(
            generator, new_prompt, discovered_data, agent_namespace, context
        )

        # Sanitize environment to hide secrets
        env_backup = context.sanitize_environment()

        try:
            # Execute the generated code
            logger.info("Executing generated agent code")
            exec(code, agent_namespace)
            result = agent_namespace.get("result", True)
            logger.info("Agent execution completed successfully")

        finally:
            # Restore environment
            context.restore_environment(env_backup)

        # Return result
        faasr_return(result)

    except Exception as e:
        err_msg = f"Agent execution failed: {str(e)}"
        traceback = tb_module.format_exc()
        logger.error(f"{err_msg}\n{traceback}")
        faasr_exit(message=err_msg, traceback=traceback)


def _log_generated_code_to_s3(faasr, action_name, code):
    """
    Log the generated agent code to S3 for debugging/auditing
    
    Arguments:
        faasr: FaaSr payload instance
        action_name: Name of the agent action
        code: Generated Python code to log
    """
    try:
        from FaaSr_py.client.py_client_stubs import faasr_put_file
        import time
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        code_filename = f"{action_name}_generated_{timestamp}.py"
        
        # Write code to temp file
        temp_path = f"/tmp/{code_filename}"
        with open(temp_path, "w") as f:
            f.write(code)
        
        # Upload to S3
        faasr_put_file(code_filename, code_filename, local_folder="/tmp", remote_folder="agent_generated_code")
        logger.info(f"Logged generated code to S3: agent_generated_code/{code_filename}")
        
    except Exception as e:
        logger.warning(f"Could not log generated code to S3: {e}")


def _explore_s3_context(faasr, context):
    """
    Explore S3 to provide context for agent code generation
    
    Returns:
        dict: S3 exploration data including file list (if reasonable size)
    """
    from FaaSr_py.client.py_client_stubs import faasr_get_folder_list
    
    try:
        # Get list of files (limited exploration)
        files = faasr_get_folder_list()
        
        # Only include if list is reasonable size
        if len(files) <= 100:
            # Group by prefix/folder
            folders = {}
            for file in files:
                parts = file.split('/')
                if len(parts) > 1:
                    folder = parts[0]
                    if folder not in folders:
                        folders[folder] = []
                    folders[folder].append(file)
                else:
                    if 'root' not in folders:
                        folders['root'] = []
                    folders['root'].append(file)
            
            return {
                "file_count": len(files),
                "files": files[:50],  # First 50 files
                "folders": list(folders.keys()),
                "structure": folders if len(files) <= 20 else None
            }
        else:
            return {
                "file_count": len(files),
                "note": "Too many files to list individually",
                "sample_files": files[:10]
            }
    except Exception as e:
        logger.warning(f"Could not explore S3: {e}")
        return {"error": "Could not list S3 files"}


def _adaptive_regeneration(generator, new_prompt, discovered_data, namespace, context):
    """
    Allow agent to regenerate approach based on discovered data
    
    This function is exposed to agent code as _regenerate_approach()
    """
    logger.info("Agent requesting adaptive regeneration based on discoveries")
    
    # Combine original context with new discoveries
    enhanced_prompt = f"""Based on discovered data: {discovered_data}

{new_prompt}"""
    
    # Generate new code
    new_code = generator.generate_code_with_context(enhanced_prompt, discovered_data)
    
    # Validate safety
    if not generator.validate_code_safety(new_code):
        raise RuntimeError("Regenerated code failed safety validation")
    
    # Execute new code in same namespace
    exec(new_code, namespace)
    

def _prepare_agent_namespace(faasr, context):
    """
    Prepare the execution namespace for agent code

    Arguments:
        faasr: FaaSr payload
        context: AgentContextManager instance

    Returns:
        dict: Namespace for exec()
    """
    # Create safe namespace with only allowed functions
    from FaaSr_py.client.agent_stubs import (
        agent_get_file,
        agent_get_folder_list,
        agent_invocation_id,
        agent_log,
        agent_put_file,
        agent_rank,
    )

    namespace = {
        "__builtins__": _get_safe_builtins(),
        # Core FaaSr functions (with agent constraints)
        "faasr_put_file": agent_put_file,
        "faasr_get_file": agent_get_file,
        "faasr_get_folder_list": agent_get_folder_list,
        "faasr_log": agent_log,
        "faasr_invocation_id": agent_invocation_id,
        "faasr_rank": agent_rank,
        # Common imports
        "json": __import__("json"),
        "os": __import__("os"),
        "sys": __import__("sys"),
        "csv": __import__("csv"),
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "re": __import__("re"),
        "pathlib": __import__("pathlib"),
        "Path": Path,
        # Store context for validators
        "_agent_context": context,
        "_agent_validator": context.validator,
    }

    return namespace


def _get_safe_builtins():
    """
    Return a restricted set of builtins for agent execution

    Returns:
        dict: Safe builtins
    """
    safe_builtins = {
        # I/O
        "print": print,
        "input": input,
        "open": open,
        # Type constructors
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "bytes": bytes,
        "bytearray": bytearray,
        "complex": complex,
        # Utilities
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        "slice": slice,
        # Type checking
        "type": type,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "callable": callable,
        # String methods
        "chr": chr,
        "ord": ord,
        "format": format,
        "repr": repr,
        "ascii": ascii,
        "bin": bin,
        "hex": hex,
        "oct": oct,
        # Error handling
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "OSError": OSError,
        "IOError": IOError,
        "FileNotFoundError": FileNotFoundError,
        # Iteration
        "iter": iter,
        "next": next,
        # Functional
        "all": all,
        "any": any,
        "hash": hash,
        "id": id,
        # Attribute access
        "getattr": getattr,
        "setattr": setattr,
        "hasattr": hasattr,
        "delattr": delattr,
        # Import
        "__import__": __import__,
    }

    return safe_builtins
