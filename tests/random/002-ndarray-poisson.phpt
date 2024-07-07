--TEST--
NDArray::poisson
--FILE--
<?php
use NDArray as nd;

new class
{
    public function __construct()
    {
        $this->testCase1();
        $this->testCase2();
        $this->testCase3();
        $this->testCase4();
    }

    /**
     * An exception is thrown when a parameter of an invalid type is passed.
     */
    private function testCase1(): void
    {
        $dataset = self::case1DataProvider();
        foreach ($dataset as $data) {
            try {
                nd::poisson($data);
            } catch (\Throwable $t) {
                echo $t->getMessage() . PHP_EOL;
            }
        }
    }

    private static function case1DataProvider(): array
    {
        return [
            'integerIsInvalid' => 1,
            'floatIsInvalid' => 3.5,
            'stringIsInvalid' => 'test',
            'booleanIsInvalid' => true,
            'nullIsInvalid' => null,
            'objectIsInvalid' => (object) [],
        ];
    }

    /**
     * An exception is thrown when passing an array with elements of an invalid type.
     */
    private function testCase2(): void
    {
        $dataset = self::case2DataProvider();
        foreach ($dataset as $data) {
            try {
                nd::poisson([$data]);
            } catch (\Throwable $t) {
                echo $t->getMessage() . PHP_EOL;
            }
        }
    }

    private static function case2DataProvider(): array
    {
        return [
            'arrayIsInvalid' => [],
            'floatIsInvalid' => 3.5,
            'stringIsInvalid' => 'test',
            'booleanIsInvalid' => true,
            'nullIsInvalid' => null,
            'objectIsInvalid' => (object) [],
        ];
    }

    /**
     * An exception is thrown when passing an empty array.
     */
    private function testCase3(): void
        {
        try {
            nd::poisson([]);
        } catch (\Throwable $t) {
            echo $t->getMessage() . PHP_EOL;
        }
    }

    /**
     * The resulting array has the correct number of dimensions and
     * each dimension contains only integers.
     */
    private function testCase4(): void
    {
        $a = nd::poisson([4]);

        echo count($a->toArray()) . PHP_EOL;

        foreach ($a->toArray() as $el) {
            if (abs($el) == $el) {
                echo 'true' . PHP_EOL;
            }
        }

        echo PHP_EOL;

        $a = nd::poisson([4, 4]);

        foreach ($a->toArray() as $el) {
            echo count($el) . PHP_EOL;
        }

        foreach ($a->toArray() as $el) {
            foreach ($el as $subEl) {
                if (abs($subEl) == $subEl) {
                    echo 'true' . PHP_EOL;
                }
            }
        }
    }
};
?>
--EXPECT--
Invalid parameter: Shape must be an array.
Invalid parameter: Shape must be an array.
Invalid parameter: Shape must be an array.
Invalid parameter: Shape must be an array.
Invalid parameter: Shape must be an array.
Invalid parameter: Shape must be an array.
Invalid parameter: Shape elements must be integers.
Invalid parameter: Shape elements must be integers.
Invalid parameter: Shape elements must be integers.
Invalid parameter: Shape elements must be integers.
Invalid parameter: Shape elements must be integers.
Invalid parameter: Shape elements must be integers.
Invalid parameter: Expected a non-empty array.
4
true
true
true
true

4
4
4
4
true
true
true
true
true
true
true
true
true
true
true
true
true
true
true
true
