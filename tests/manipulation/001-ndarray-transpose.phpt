--TEST--
NDArray::transpose
--FILE--
<?php
use \NDArray as nd;

$a = nd::array([[1, 2], [3, 4]]);
$b = nd::array([[1, 3, 2], [3, 4, 1]]);
$c = nd::array([1, 2, 3, 4]);
$d = nd::array([[[1, 2, 3, 4]]]);

print_r(nd::transpose($a)->toArray());
print_r(nd::transpose([[1, 2], [3, 4]])->toArray());
print_r(nd::transpose($b)->toArray());
print_r(nd::transpose($c)->toArray());
print_r(nd::transpose($d)->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 3
        )

    [1] => Array
        (
            [0] => 2
            [1] => 4
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 3
        )

    [1] => Array
        (
            [0] => 2
            [1] => 4
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 3
        )

    [1] => Array
        (
            [0] => 3
            [1] => 4
        )

    [2] => Array
        (
            [0] => 2
            [1] => 1
        )

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
            [0] => Array
                (
                    [0] => 1
                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => 2
                )

        )

    [2] => Array
        (
            [0] => Array
                (
                    [0] => 3
                )

        )

    [3] => Array
        (
            [0] => Array
                (
                    [0] => 4
                )

        )

)
