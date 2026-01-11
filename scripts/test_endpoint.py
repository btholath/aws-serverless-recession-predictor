#!/usr/bin/env python3
"""
Test SageMaker Endpoint with various scenarios
"""
import boto3

ENDPOINT_NAME = "fred-serverless-endpoint"
REGION = "us-east-1"

runtime = boto3.client('sagemaker-runtime', region_name=REGION)


def predict(cpi, fedfunds, t10y2y, indpro):
    """Make a single prediction"""
    payload = f"{cpi}, {fedfunds}, {t10y2y}, {indpro}"

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='text/csv',
        Body=payload
    )

    result = response['Body'].read().decode()
    return float(result.strip('[]'))


def test_endpoint():
    """Run comprehensive tests"""
    print("=" * 60)
    print("🧪 FRED Model Endpoint Tests")
    print("=" * 60)

    # Test 1: Basic connectivity
    print("\n📡 Test 1: Basic Connectivity")
    try:
        result = predict(250.0, 5.25, -0.5, 102.5)
        print(f"   ✅ Endpoint responding: {result:.2f}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return

    # Test 2: Different economic scenarios
    print("\n📊 Test 2: Economic Scenarios")

    scenarios = [
        {
            "name": "Current Economy (Baseline)",
            "cpi": 250.0, "fedfunds": 5.25, "t10y2y": -0.5, "indpro": 102.5
        },
        {
            "name": "High Inflation",
            "cpi": 300.0, "fedfunds": 7.0, "t10y2y": -1.0, "indpro": 100.0
        },
        {
            "name": "Low Rates (Stimulus)",
            "cpi": 220.0, "fedfunds": 0.25, "t10y2y": 2.0, "indpro": 105.0
        },
        {
            "name": "Recession Signal (Inverted Yield)",
            "cpi": 260.0, "fedfunds": 5.0, "t10y2y": -2.0, "indpro": 95.0
        },
        {
            "name": "Strong Growth",
            "cpi": 240.0, "fedfunds": 3.0, "t10y2y": 1.5, "indpro": 110.0
        },
    ]

    print(f"\n   {'Scenario':<35} {'Unemployment':>12}")
    print("   " + "-" * 50)

    for s in scenarios:
        result = predict(s['cpi'], s['fedfunds'], s['t10y2y'], s['indpro'])
        print(f"   {s['name']:<35} {result:>10.2f}%")

    # Test 3: Response time
    print("\n⏱️  Test 3: Response Time (5 calls)")
    import time
    times = []
    for i in range(5):
        start = time.time()
        predict(250.0, 5.25, -0.5, 102.5)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"   Call {i + 1}: {elapsed:.0f}ms")

    print(f"\n   Average: {sum(times) / len(times):.0f}ms")
    print(f"   Min: {min(times):.0f}ms | Max: {max(times):.0f}ms")

    # Test 4: Edge cases
    print("\n🔬 Test 4: Edge Cases")

    edge_cases = [
        ("Very high CPI", 400.0, 5.0, 0.0, 100.0),
        ("Zero Fed rate", 250.0, 0.0, 2.0, 100.0),
        ("Deep inversion", 250.0, 5.0, -3.0, 100.0),
        ("Low production", 250.0, 5.0, 0.0, 80.0),
    ]

    for name, cpi, ff, t10, ind in edge_cases:
        try:
            result = predict(cpi, ff, t10, ind)
            print(f"   ✅ {name}: {result:.2f}%")
        except Exception as e:
            print(f"   ❌ {name}: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_endpoint()
