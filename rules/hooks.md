# Hooks Guidelines

Rules for using Claude Code hooks in ML workflows.

## Available Hooks

### PreToolUse
Runs before tool execution:
- GPU availability check before training
- Package installation reminders
- tmux reminder for long runs

### PostToolUse
Runs after tool execution:
- Auto-format Python with ruff
- Type check with mypy
- Warn about print statements

### SessionStart
Runs at session start:
- Display Python/PyTorch/RDKit versions
- Load environment info

### Stop
Runs when Claude responds:
- Check modified files

## Hook Behavior

### GPU Check (PreToolUse)

Triggered when: Running training commands (`python train.py`)

Output:
```
[Hook] GPU available: True
[Hook] Device count: 1
[Hook] GPU 0: NVIDIA A100
```

### Auto-Format (PostToolUse)

Triggered when: Editing `.py` files

Action: Runs `ruff check --fix` automatically

### Type Check (PostToolUse)

Triggered when: Editing `.py` files

Output: First 15 lines of mypy output for that file

### Print Warning (PostToolUse)

Triggered when: Editing `.py` files containing `print()`

Output:
```
[Hook] WARNING: print() at line 45: print(f"Debug: {value}")
```

## Customizing Hooks

Edit `hooks/hooks.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "tool == \"Edit\" && tool_input.file_path matches \"\\.py$\"",
        "hooks": [
          {
            "type": "command",
            "command": "ruff check --fix \"${tool_input.file_path}\""
          }
        ]
      }
    ]
  }
}
```

## Matcher Syntax

```
tool == "Bash"                           # Tool type
tool_input.command matches "python.*"    # Command pattern
tool_input.file_path matches "\\.py$"    # File pattern
```

## Best Practices

1. **Keep hooks fast** - Long-running hooks block execution
2. **Use async for slow tasks** - Background processing
3. **Don't block critical operations** - Use warnings, not blocks
4. **Log hook actions** - Prefix output with `[Hook]`

## Debugging Hooks

Check hook execution:
```bash
# In hooks.json, add verbose output
"command": "echo '[Hook] Running check...' && mypy file.py"
```

## Session Hooks

### SessionStart
Good for:
- Environment verification
- Loading context
- Displaying versions

### SessionEnd
Good for:
- Saving state
- Cleanup
- Logging session summary

## Hook Limitations

- Hooks run synchronously by default
- Large outputs are truncated
- Hooks can't modify Claude's responses
- Failed hooks don't block tool execution (unless exit code non-zero)
