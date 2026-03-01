def write_iteration_log(
    *,
    filepath: str,
    code_before_feedback: str,
    final_code: str,
    output: str,
    error: str,
) -> None:
    """Write a single run log file with consistent section headers."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("############################################################################\n")
        f.write("### CODE BEFORE FEEDBACK\n")
        f.write("############################################################################\n\n")
        f.write(str(code_before_feedback) + "\n\n\n")

        f.write("############################################################################\n")
        f.write("### FINAL CODE\n")
        f.write("############################################################################\n\n")
        f.write(str(final_code) + "\n\n\n")

        f.write("############################################################################\n")
        f.write("### OUTPUT\n")
        f.write("############################################################################\n\n")
        f.write(str(output) + "\n\n\n")

        f.write("############################################################################\n")
        f.write("### ERROR\n")
        f.write("############################################################################\n\n")
        f.write(str(error) + "\n\n")