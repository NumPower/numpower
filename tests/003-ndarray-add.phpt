--TEST--
NDArray::add
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(($a + 2)->toArray());
print_r(($a + $a)->toArray());
print_r(($a + $a[0])->toArray());
print_r(($a + [[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
        )

    [1] => Array
        (
            [0] => 6
            [1] => 8
        )

)
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
        )

    [1] => Array
        (
            [0] => 6
            [1] => 8
        )

)
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
        )

    [1] => Array
        (
            [0] => 4
            [1] => 6
        )

)
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 3
        )

    [1] => Array
        (
            [0] => 5
            [1] => 6
        )

)