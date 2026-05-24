# Contributing to Thurstone

## Development Standards

### Code Quality Requirements

**MANDATORY - All code must pass these checks before any PR:**

```bash
# 1. All tests must pass
python -m pytest tests/ -v

# 2. Core package must have ZERO linting issues
flake8 thurstone/ --count --max-complexity=25 --max-line-length=127 --statistics

# 3. Code must be properly formatted
black --check .
isort --check-only .

# 4. NO EMOJIS anywhere in code (CI will reject)
# Use plain text for all messages and comments
```

### Branch Management Rules

#### **THE GOLDEN RULES:**

1. **Only one new branch at a time**
   - Finish and merge current work before starting new branch
   - Prevents conflicts and confusion
   - Keeps git history clean

2. **Always rebase to main**
   - `git rebase origin/main` before any PR
   - No merge commits cluttering history
   - Linear, clean commit sequence

3. **Always test and lint before pushing**
   - Every single commit must pass all checks
   - No exceptions, no "fix later" commits
   - Quality gate is non-negotiable

#### **Workflow:**
```bash
# 1. Start from clean main
git checkout main
git pull origin main

# 2. Create single feature branch
git checkout -b feature/your-change

# 3. Do your work, test continuously
# Edit code...
python -m pytest tests/
flake8 thurstone/ --count --max-complexity=25 --max-line-length=127
black . && isort .

# 4. Before PR: rebase to main
git rebase origin/main

# 5. Final verification
python -m pytest tests/
flake8 thurstone/ --count --max-complexity=25 --max-line-length=127

# 6. Push and PR
git push origin feature/your-change
```

### Pull Request Process

#### **Before Opening a PR:**

1. **Run all quality checks:**
   ```bash
   # Format code
   black .
   isort .
   
   # Run tests
   python -m pytest tests/
   
   # Check linting (core package only)
   flake8 thurstone/ --count --max-complexity=25 --max-line-length=127
   
   # Verify no emojis (use common emoji patterns)
   grep -r "[[:emoji:]]" thurstone/ && echo "EMOJIS FOUND" || echo "No emojis detected"
   ```

2. **Write clear PR description:**
   - What problem does this solve?
   - What changes were made?
   - How was it tested?
   - Breaking changes?

3. **Ensure CI will pass:**
   - All tests pass locally
   - Core package linting is clean
   - No emojis in code

#### **PR Review Standards:**

- **Code must be production-ready**
- **All edge cases considered**
- **Adequate test coverage**
- **Documentation updated if needed**
- **No temporary/debug code**

### Development Environment Setup

#### **Required Tools:**
```bash
# Install formatting tools
pipx install black
pipx install isort
pipx install flake8

# Install package in development mode
pip install -e ".[test]"
```

#### **Pre-commit Checks (Recommended):**
```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running pre-commit checks..."

# Format code
black .
isort .

# Check core package linting
if ! flake8 thurstone/ --count --max-complexity=25 --max-line-length=127 --statistics; then
    echo "ERROR: Core package linting failed"
    exit 1
fi

# Check for emojis (use common emoji patterns)
if grep -r "[[:emoji:]]" thurstone/; then
    echo "Emojis found in core package"
    exit 1
fi

# Run tests
if ! python -m pytest tests/ -q; then
    echo "ERROR: Tests failed"
    exit 1
fi

echo "SUCCESS: All pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit
```

## Core Package vs Examples

### **Strict Standards Apply To:**
- `thurstone/` - Core library package
- `tests/` - Test suite
- Root configuration files

### **Relaxed Standards For:**
- `examples/` - Demo scripts (allowed to have linting issues)
- `research/` - Research code (exploratory, not production)
- `scripts/` - Utility scripts

**Why:** Users import the core package, so it must be pristine. Examples are for learning and can prioritize clarity over perfect style.

## Common Mistakes to Avoid

### **ERROR: Don't Do This:**

1. **Working on multiple branches simultaneously without coordination**
   - Creates merge conflicts
   - Lost work risk
   - Confusing git history

2. **Ignoring linting in core package**
   - CI will fail
   - Code quality degrades
   - Technical debt accumulates

3. **Using emojis anywhere in code**
   - Unprofessional appearance
   - CI enforcement will reject
   - Accessibility issues

4. **Committing without testing**
   - Broken main branch
   - CI failures
   - Wasted reviewer time

5. **Large, unfocused PRs**
   - Hard to review
   - Increases conflict risk
   - Harder to revert if needed

### **✅ Do This Instead:**

1. **Coordinated development:**
   - Communicate about branch work
   - Create backup branches for safety
   - Merge small, focused changes frequently

2. **Quality-first approach:**
   - Run all checks before committing
   - Fix linting issues immediately
   - Write tests for new features

3. **Professional presentation:**
   - Clear, emoji-free code
   - Descriptive variable names
   - Proper documentation

## Emergency Procedures

### **If You Break Main:**
```bash
# 1. Immediately create issue
# 2. Revert the breaking commit
git revert <commit-hash>
git push origin main

# 3. Fix on a branch, test thoroughly, then PR
```

### **If You Lose Work:**
```bash
# Check for backup branches
git branch | grep backup

# Check reflog for lost commits  
git reflog

# Recover from remote if needed
git fetch origin
```

### **If Branches Diverge:**
```bash
# Create safety backups FIRST
git branch backup-current-work

# Then attempt merge
git merge origin/main
# Resolve conflicts carefully
# Test everything before pushing
```

## Code Review Guidelines

### **As a Reviewer:**
- ✅ Does it solve the stated problem?
- ✅ Are all tests passing?
- ✅ Is the core package lint-clean?
- ✅ Is it properly documented?
- ✅ Are there any emoji violations?
- ✅ Is the approach sound?

### **As a Contributor:**
- ✅ Include context in PR description
- ✅ Respond to feedback promptly
- ✅ Test reviewer suggestions
- ✅ Keep PR scope focused

## Release Process

1. **All PRs reviewed and merged to main**
2. **Full test suite passing**
3. **Core package 100% lint-clean**
4. **Documentation updated**
5. **Version bumped in pyproject.toml**
6. **Release notes prepared**

## Questions?

For any questions about these guidelines:
1. Check existing issues/PRs for similar situations
2. Create an issue for clarification
3. Follow the established patterns in the codebase

**Remember: Better to ask than to guess and create technical debt.**

---

*These guidelines exist to maintain code quality and prevent the kinds of issues that waste development time. Following them makes everyone more productive.*