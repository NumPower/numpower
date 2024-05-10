--TEST--
NDArray::reshape
--FILE--
<?php
use \NDArray as nd;

$a = nd::array([[1, 2], [3, 4]]);
$b = nd::array([1, 2, 3, 4]);

print_r(nd::reshape($a, [4])->toArray());
print_r(nd::reshape([[1, 2], [3, 4]], [4])->toArray());
print_r(nd::reshape($a, [1, 4])->toArray());
print_r(nd::reshape($a, [1, 2, 2])->toArray());
print_r(nd::reshape($b, [2, 2])->toArray());
print_r(nd::reshape(nd::reshape($b, [2, 2]), [1, 4])->toArray());
?>
--EXPECT--
Array
(
    [0] => 1
    [1] => 2
    [2] => 3
    [3] => 4
)
Array
(
    [0] => 1
    [1] => 2
    [2] => 3
    [3] => 4
)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
            [2] => 3
            [3] => 4
        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 1
                    [1] => 2
                )

            [1] => Array
                (
                    [0] => 3
                    [1] => 4
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
        )

    [1] => Array
        (
            [0] => 3
            [1] => 4
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
            [2] => 3
            [3] => 4
        )

)