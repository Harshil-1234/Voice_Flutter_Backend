#!/bin/bash
# Quick verification script for GCP VM - run this to diagnose LocalLLMService issues
# Usage: bash verify_llm.sh

echo "🔍 Checking LocalLLMService on GCP VM..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS="${GREEN}✅${NC}"
FAIL="${RED}❌${NC}"
WARN="${YELLOW}⚠️${NC}"

# Check 1: Python version
echo -n "1. Python version: "
if python3 --version &>/dev/null; then
    echo -e "${PASS} $(python3 --version)"
else
    echo -e "${FAIL} Python3 not found"
    exit 1
fi

# Check 2: Venv activation
echo -n "2. Virtual environment: "
if [ -f ~/Voice_Flutter_Backend/venv/bin/activate ]; then
    echo -e "${PASS} Found"
    source ~/Voice_Flutter_Backend/venv/bin/activate
else
    echo -e "${FAIL} Not found"
    exit 1
fi

# Check 3: Disk space
echo -n "3. Disk space: "
FREE_GB=$(df /home | awk 'NR==2 {print $4/1024/1024}')
if (( $(echo "$FREE_GB > 3" | bc -l) )); then
    echo -e "${PASS} ${FREE_GB:.1f} GB free"
else
    echo -e "${WARN} ${FREE_GB:.1f} GB free (need >3GB)"
fi

# Check 4: RAM
echo -n "4. Available RAM: "
FREE_RAM=$(free | awk 'NR==2 {print $7/1024/1024}')
if (( $(echo "$FREE_RAM > 2" | bc -l) )); then
    echo -e "${PASS} ${FREE_RAM:.1f} GB free"
else
    echo -e "${WARN} ${FREE_RAM:.1f} GB free (need >6GB total)"
fi

# Check 5: llama-cpp-python
echo -n "5. llama-cpp-python: "
if python3 -c "from llama_cpp import Llama" 2>/dev/null; then
    echo -e "${PASS} Installed"
else
    echo -e "${FAIL} NOT installed"
    echo "   Fix: pip install llama-cpp-python==0.3.5"
fi

# Check 6: huggingface-hub
echo -n "6. huggingface-hub: "
if python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null; then
    echo -e "${PASS} Installed"
else
    echo -e "${FAIL} NOT installed"
    echo "   Fix: pip install huggingface-hub"
fi

# Check 7: Model cache
echo -n "7. Model cache: "
if [ -f ~/.cache/llm_models/Gemma-2-2b-it-Q4_K_M.gguf ]; then
    SIZE=$(ls -lh ~/.cache/llm_models/Gemma-2-2b-it-Q4_K_M.gguf | awk '{print $5}')
    echo -e "${PASS} ${SIZE}"
else
    echo -e "${WARN} Not downloaded yet (will download on first run)"
fi

# Check 8: Service status
echo -n "8. Service status: "
if sudo systemctl is-active voice-backend.service &>/dev/null; then
    echo -e "${PASS} Running"
else
    echo -e "${WARN} Not running (expected if doing manual test)"
fi

# Check 9: Direct Python test
echo ""
echo "9. Testing LocalLLMService directly..."
cd ~/Voice_Flutter_Backend

python3 << 'PYEOF'
import sys
try:
    from app.LocalLLMService import get_local_llm_service
    print("   [Loading model...]")
    service = get_local_llm_service()
    print("   ✅ SUCCESS: LocalLLMService initialized!")
    print("   Service is ready for inference.")
except ImportError as e:
    print(f"   ❌ Import Error: {e}")
    print("   Fix: pip install llama-cpp-python huggingface-hub")
    sys.exit(1)
except MemoryError as e:
    print(f"   ❌ Memory Error: Insufficient RAM")
    print("   Need: 6-8GB RAM, Current: check with 'free -h'")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"   ❌ File Not Found: {e}")
    print("   Model not downloaded. Starting download (takes 5-10 min)...")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {type(e).__name__}: {e}")
    import traceback
    print("\n   Traceback:")
    traceback.print_exc()
    sys.exit(1)
PYEOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ ALL CHECKS PASSED!"
    echo "=================================="
    echo ""
    echo "You can now:"
    echo "1. Restart service: sudo systemctl restart voice-backend.service"
    echo "2. Check logs: sudo journalctl -u voice-backend.service -f"
    echo "3. Verify: curl http://localhost:8000/health"
else
    echo ""
    echo "=================================="
    echo "❌ CHECKS FAILED - See errors above"
    echo "=================================="
    echo ""
    echo "For detailed troubleshooting, see GCP_TROUBLESHOOT.md"
fi
