--TEST--
NDArray::standard_normal
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
        $this->testCase5();
    }

    /**
     * Successful creation.
     */
    private function testCase1(): void
    {
        echo '--- CASE 1 ---' . PHP_EOL;
        $dataset = self::case1DataProvider();
        foreach ($dataset as $sk => $shape) {
            nd::standard_normal($shape);
            echo "successful creation with $sk" . PHP_EOL;
        }
        echo PHP_EOL;
    }

    private static function case1DataProvider(): array
    {
        return [
            '1-dim' => [1],
            '2-dim' => [2, 3],
            '3-dim' => [2, 3, 4],
        ];
    }

    /**
     * An exception is thrown when parameter has an invalid type.
     */
    private function testCase2(): void
    {
        echo '--- CASE 2 ---' . PHP_EOL;
        $dataset = self::case2DataProvider();
        foreach ($dataset as $condition => $data) {
            try {
                nd::standard_normal($data);
            } catch (\Throwable $t) {
                echo "Error when passed " . $condition. ": " . $t->getMessage() . PHP_EOL;
            }
        }
        echo PHP_EOL;
    }

    private static function case2DataProvider(): array
    {
        return [
            'shape is integer' => 1,
            'shape is double' => 3.5,
            'shape is string' => 'test',
            'shape is boolean' => true,
            'shape is null' => null,
            'shape is object' => (object) [],
        ];
    }

    /**
     * An exception is thrown when passing an array with elements of an invalid type.
     */
    private function testCase3(): void
    {
        echo '--- CASE 3 ---' . PHP_EOL;
        $dataset = self::case3DataProvider();
        foreach ($dataset as $condition =>  $data) {
            try {
                nd::standard_normal([$data]);
            } catch (\Throwable $t) {
                echo "Error when shape " . $condition . ": " . $t->getMessage() . PHP_EOL;
            }
        }
        echo PHP_EOL;
    }

    private static function case3DataProvider(): array
    {
        return [
            'value is array' => [],
            'value is float' => 3.5,
            'value is string' => 'test',
            'value is boolean' => true,
            'value is null' => null,
            'value is object' => (object) [],
        ];
    }

    /**
     * An exception is thrown when passing an empty array.
     */
    private function testCase4(): void
    {
        echo '--- CASE 4 ---' . PHP_EOL;
        try {
            nd::standard_normal([]);
        } catch (\Throwable $t) {
            echo $t->getMessage() . PHP_EOL;
        }
        echo PHP_EOL;
    }

    /**
     * The resulting array has the correct number of dimensions and
     * each dimension contains only integers.
     */
    private function testCase5(): void
    {
        echo '--- CASE 5 ---' . PHP_EOL;
        $a = nd::standard_normal([4]);

        echo "Shape is: " . count($a->toArray()) . PHP_EOL;

        foreach ($a->toArray() as $elk => $el) {
            if (is_float($el)) {
                echo "element " . $elk + 1 . " is double" . PHP_EOL;
            }
        }

        $a = nd::standard_normal([4, 4]);

        foreach ($a->toArray() as $k => $el) {
            echo "Shape level of element " . $k + 1 . " is: " . count($el) . PHP_EOL;
        }

        foreach ($a->toArray() as $elk => $el) {
            foreach ($el as $selk => $subEl) {
                if (is_float($subEl)) {
                    echo "sub-element " . $selk . " of element " . $elk + 1 . " is double" . PHP_EOL;
                }
            }
        }
    }
};
?>
--EXPECT--
--- CASE 1 ---
successful creation with 1-dim
successful creation with 2-dim
successful creation with 3-dim

--- CASE 2 ---
Error when passed shape is integer: NDArray::standard_normal(): Argument #1 ($shape) must be of type array, int given
Error when passed shape is double: NDArray::standard_normal(): Argument #1 ($shape) must be of type array, float given
Error when passed shape is string: NDArray::standard_normal(): Argument #1 ($shape) must be of type array, string given
Error when passed shape is boolean: NDArray::standard_normal(): Argument #1 ($shape) must be of type array, true given
Error when passed shape is null: NDArray::standard_normal(): Argument #1 ($shape) must be of type array, null given
Error when passed shape is object: NDArray::standard_normal(): Argument #1 ($shape) must be of type array, stdClass given

--- CASE 3 ---
Error when shape value is array: Invalid parameter: Shape elements must be integers.
Error when shape value is float: Invalid parameter: Shape elements must be integers.
Error when shape value is string: Invalid parameter: Shape elements must be integers.
Error when shape value is boolean: Invalid parameter: Shape elements must be integers.
Error when shape value is null: Invalid parameter: Shape elements must be integers.
Error when shape value is object: Invalid parameter: Shape elements must be integers.

--- CASE 4 ---
Invalid parameter: Expected a non-empty array.

--- CASE 5 ---
Shape is: 4
element 1 is double
element 2 is double
element 3 is double
element 4 is double
Shape level of element 1 is: 4
Shape level of element 2 is: 4
Shape level of element 3 is: 4
Shape level of element 4 is: 4
sub-element 0 of element 1 is double
sub-element 1 of element 1 is double
sub-element 2 of element 1 is double
sub-element 3 of element 1 is double
sub-element 0 of element 2 is double
sub-element 1 of element 2 is double
sub-element 2 of element 2 is double
sub-element 3 of element 2 is double
sub-element 0 of element 3 is double
sub-element 1 of element 3 is double
sub-element 2 of element 3 is double
sub-element 3 of element 3 is double
sub-element 0 of element 4 is double
sub-element 1 of element 4 is double
sub-element 2 of element 4 is double
sub-element 3 of element 4 is double
