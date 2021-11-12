---
redirect_from: /_posts/2021-9-15-Data_Structure.md/
title: DataStructure
tags:
  - Java
  - DataStructure
---

# Data Structure

- [Data Structure](#data-structure)
  - [Array](#array)
    - [稀疏数组](#稀疏数组)
    - [时间复杂度](#时间复杂度)
  - [Linked List](#linked-list)
    - [单链表](#单链表)
      - [结点结构](#结点结构)
      - [单链表实现 ](#单链表实现-)
    - [双指针技巧](#双指针技巧)
      - [代码模板](#代码模板)
      - [注意事项](#注意事项)
      - [复杂度分析](#复杂度分析)
    - [环形链表](#环形链表)
      - [LeetCode 141 Linked List Cycle](#leetcode-141-linked-list-cycle)
        - [代码实现](#代码实现)
    - [环形链表II](#环形链表ii)
      - [LeetCode 142 Linked List Cycle II](#leetcode-142-linked-list-cycle-ii)
        - [代码实现](#代码实现-1)
    - [相交链表](#相交链表)
      - [LeetCode 160 相交链表](#leetcode-160-相交链表)
        - [代码实现](#代码实现-2)
    - [删除链表的倒数第N个节点](#删除链表的倒数第n个节点)
        - [代码实现](#代码实现-3)
    - [反转链表](#反转链表)
      - [代码实现](#代码实现-4)
    - [移除链表元素](#移除链表元素)
      - [代码实现](#代码实现-5)
    - [奇偶链表](#奇偶链表)
      - [LeetCode](#leetcode)
        - [代码实现](#代码实现-6)
    - [回文链表](#回文链表)
      - [LeetCode 234 回文链表](#leetcode-234-回文链表)
        - [代码实现](#代码实现-7)
    - [小结 链表经典问题](#小结-链表经典问题)
    - [双链表](#双链表)
      - [结点结构](#结点结构-1)
      - [添加操作](#添加操作)
      - [删除操作](#删除操作)
      - [双链表实现](#双链表实现)
    - [时间复杂度比较](#时间复杂度比较)
    - [合并两个有序链表](#合并两个有序链表)
      - [Leetcode 21 Merge Two Sorted Lists ](#leetcode-21-merge-two-sorted-lists-)
        - [代码实现](#代码实现-8)
    - [两数相加](#两数相加)
      - [Leetcode 2 addTwoNumbers](#leetcode-2-addtwonumbers)
        - [代码实现](#代码实现-9)
    - [ 扁平化多级双向链表](#-扁平化多级双向链表)
      - [LeetCode 430 Flatten a Multilevel Doubly Linked List](#leetcode-430-flatten-a-multilevel-doubly-linked-list)
        - [代码实现](#代码实现-10)
    - [复制带随机指针的链表](#复制带随机指针的链表)
      - [LeetCode 138 Copy List with Random Pointer](#leetcode-138-copy-list-with-random-pointer)
        - [代码实现](#代码实现-11)
    - [旋转链表](#旋转链表)
      - [LeetCode 61 Rotate List ](#leetcode-61-rotate-list-)
        - [代码实现](#代码实现-12)
  - [Stack](#stack)
  - [queue](#queue)
    - [队列实现](#队列实现)
  - [binary-tree](#binary-tree)
  - [binary-search-tree](#binary-search-tree)
  - [balanced-search-tree](#balanced-search-tree)
    - [介绍](#介绍)
    - [AVL 树](#avl-树)
  - [heap](#heap)
  - [trie](#trie)
  - [union-find](#union-find)
  - [hash-table](#hash-table)

## Array

### 稀疏数组

一个经典的demo：

```java
package d2j.jse.datastructures;

public class SparseArray {
    public static void main(String[] args) {
        //以棋盘为例
        //创建一个原始的二维数组 11*11
        //0:没有棋子；1:黑；2:白
        int[][] chessArr1 = new int[11][11];
        chessArr1[1][2] = 1;
        chessArr1[2][3] = 2;
        chessArr1[5][4] = 2;
        System.out.println("原始的二维数组：");
        for(int[] row : chessArr1){
            for (int data : row){
                System.out.printf("%d\t", data);
            }
            System.out.println();
        }
        //二维数组转稀疏数组
        //1.先遍历二维数组 得到非0数据的个数
        int sum = 0;
        for (int[] ints : chessArr1) {
            for (int j = 0; j < chessArr1.length; j++) {
                if (ints[j] != 0) {
                    sum++;
                }
            }
        }
        //2.创建对应的稀疏数组
        int[][] sparseArr = new int[sum + 1][3];
        //给系数数组赋值
        sparseArr[0][0] = 11;
        sparseArr[0][1] = 11;
        sparseArr[0][2] = sum;
        //遍历二维数组，将非0的值存放到sparseArr中
        int k=0;
        for (int i = 0; i < chessArr1.length; i++) {
            for (int j = 0; j < chessArr1.length; j++) {
                if(chessArr1[i][j] != 0)
                {
                    k++;
                    sparseArr[k][0] = i;
                    sparseArr[k][1] = j;
                    sparseArr[k][2] = chessArr1[i][j];
                }
            }
        }
        //输出稀疏数组
        System.out.println("\n得到的稀疏数组为：");
        for (int[] i:sparseArr) {
            System.out.printf("%d\t%d\t%d\n",i[0],i[1],i[2]);
        }
        //稀疏数组转二维数组
        //先读取稀疏数组的第一行，创建二维数组
        int[][] chessArr2 = new int[sparseArr[0][0]][sparseArr[0][1]];
        for (int[] i:sparseArr) {
            if(i[0]!=sparseArr[0][0])
            {
                chessArr2[i[0]][i[1]] = i[2];
            }
        }
        System.out.println("\n恢复后的二维数组：");
        for(int[] row : chessArr2){
            for (int data : row){
                System.out.printf("%d\t", data);
            }
            System.out.println();
        }
    }
}

```

### 时间复杂度

- 在数组的**末尾插入/删除**、**更新**、**获取**某个位置的元素，都是 O(1) 的时间复杂度
- 在数组的任何其它地方**插入/删除**元素，都是 O(n) 的时间复杂度
- 空间复杂度：O(n)



## Linked List

与数组相似，链表也是一种线性数据结构。这里有一个例子：
<img src="https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/12/screen-shot-2018-04-12-at-152754.png" style="zoom:60%;"/>
正如你所看到的，链表中的每个元素实际上是一个单独的对象，而所有对象都通过每个元素中的引用字段链接在一起。
链表有两种类型：单链表和双链表。上面给出的例子是一个单链表，这里有一个双链表的例子：

<img src="https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/17/screen-shot-2018-04-17-at-161130.png" style="zoom:60%;" />

### 单链表

单链表中的每个结点不仅包含值，还包含链接到下一个结点的`引用字段`。通过这种方式，单链表将所有结点按顺序组织起来。如上图：蓝色箭头显示单个链接列表中的结点是如何组合在一起的。

#### 结点结构

```java
// Definition for singly-linked list.
public class SinglyListNode {
    int val;
    SinglyListNode next;
    SinglyListNode(int x) { val = x; }
}
```

在大多数情况下，使用头结点(第一个结点)来表示整个列表。

#### 单链表实现 

> LeetCode 707 Design Linked List
>
> 设计链表的实现。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。
>
> 在链表类中实现这些功能：
>
> `get(index)`：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
> `addAtHead(val)`：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
> `addAtTail(val)`：将值为 val 的节点追加到链表的最后一个元素。
> `addAtIndex(index,val)`：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
> `deleteAtIndex(index)`：如果索引 index 有效，则删除链表中的第 index 个节点。

```java
class Node{
    public int val;
    public Node next;
    public Node(int val){
        this.val = val;
        this.next = null;
    }
}

class MyLinkedList {
    Node head = null;
    int size;
    /** Initialize your data structure here. */
    public MyLinkedList() {
        this.size = 0;
        this.head = null;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int get(int index) {
        if(index<0||index>=size)
        {
            return -1;
        }
        Node tmp = head;
        for(int i=1;i<=index;i++)
        {
            tmp=tmp.next;
        }
        return tmp.val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void addAtHead(int val) {
        Node p = new Node(val);
        p.next = head;
        head = p;
        size++;
    }
    
    /** Append a node of value val to the last element of the linked list. */
    public void addAtTail(int val) {
        Node p = new Node(val);
        if(size==0)
        {
            head = p;
            size++;
            return;
        }
        else
        {
            Node tmp = head;
            while(tmp.next!=null)
            {
                tmp=tmp.next;
            }
            
            tmp.next=p;
            p.next=null;
            size++;
        }
        
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void addAtIndex(int index, int val) {
        Node p = new Node(val);
        if(index==size){
            addAtTail(val);
            return;
        }
        else if(index<=0)
        {
            addAtHead(val);
            return;
        }
        else if(index>size)
        {
            return;
        }
        Node tmp=head;
        for(int i=1;i<index;i++)
        {
            tmp=tmp.next;
        }
        p.next = tmp.next;
        tmp.next = p;
        size++;
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
        if(index<0||index>=size)
        {
            return;
        }
        if(index==0)
        {
            head=head.next;
            size--;
            return;
        }
        Node tmp=head;
        for(int i=1;i<index;i++)
        {
            tmp=tmp.next;
        }
        tmp.next = tmp.next.next;
        size--;
    }
}

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.get(index);
 * obj.addAtHead(val);
 * obj.addAtTail(val);
 * obj.addAtIndex(index,val);
 * obj.deleteAtIndex(index);
 */
```

### 双指针技巧

> 给定一个链表，判断链表中是否有环。

想象一下，有两个速度不同的跑步者。如果他们在直路上行驶，快跑者将首先到达目的地。但是，如果它们在圆形跑道上跑步，那么快跑者如果继续跑步就会追上慢跑者。

这正是我们在链表中使用两个速度不同的指针时会遇到的情况：

1. 如果没有环，快指针将停在链表的末尾。
2. 如果有环，快指针最终将与慢指针相遇。

> 这两个指针的适当速度应该是多少？

一个安全的选择是每次移动慢指针一步，而移动快指针两步。每一次迭代，快速指针将额外移动一步。如果环的长度为 M，经过 M 次迭代后，快指针肯定会多绕环一周，并赶上慢指针。

#### 代码模板

```java
// Initialize slow & fast pointers
ListNode slow = head;
ListNode fast = head;
/**
 * Change this condition to fit specific problem.
 * Attention: remember to avoid null-pointer error
 **/
while (slow != null && fast != null && fast.next != null) {
    slow = slow.next;           // move slow pointer one step each time
    fast = fast.next.next;      // move fast pointer two steps each time
    if (slow == fast) {         // change this condition to fit specific problem
        return true;
    }
}
return false;   // change return value to fit specific problem
```

```c++
// Initialize slow & fast pointers
ListNode* slow = head;
ListNode* fast = head;
/**
 * Change this condition to fit specific problem.
 * Attention: remember to avoid null-pointer error
 **/
while (slow && fast && fast->next) {
    slow = slow->next;          // move slow pointer one step each time
    fast = fast->next->next;    // move fast pointer two steps each time
    if (slow == fast) {         // change this condition to fit specific problem
        return true;
    }
}
return false;   // change return value to fit specific problem
```

#### 注意事项

1.  **在调用 next 字段之前，始终检查节点是否为空。**

获取空节点的下一个节点将导致空指针错误。例如，在我们运行 fast = fast.next.next 之前，需要检查 fast 和 fast.next 不为空。

2.  **仔细定义循环的结束条件。**

#### 复杂度分析

空间复杂度分析容易。如果只使用指针，而不使用任何其他额外的空间，那么空间复杂度将是 O(1)。但是，时间复杂度的分析比较困难。为了得到答案，我们需要分析运行循环的次数。

在前面的查找循环示例中，假设我们每次移动较快的指针 2 步，每次移动较慢的指针 1 步。

1. 如果没有循环，快指针需要 N/2 次才能到达链表的末尾，其中 N 是链表的长度。
2. 如果存在循环，则快指针需要 M 次才能赶上慢指针，其中 M 是列表中循环的长度。

显然，M <= N 。所以我们将循环运行 N 次。对于每次循环，我们只需要常量级的时间。因此，该算法的时间复杂度总共为 O(N)。



### 环形链表

#### LeetCode 141 Linked List Cycle

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

要求： *O(1)*（即，常量）内存解决

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

**输入：**head = [3,2,0,-4], pos = 1
**输出：**true
**解释：**链表中有一个环，其尾部连接到第二个节点。

**提示：**

- 链表中节点的数目范围是 `[0, 10^4]`
- `-10^5 <= Node.val <= 10^5`
- `pos` 为 `-1` 或者链表中的一个 **有效索引** 。

##### 代码实现

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        try {
            while(fast.next.next!=null)
            {
                head=head.next;
                fast=fast.next.next;
                if(fast.next==head.next)
                {
                    return true;
                }
            }
        } catch (NullPointerException e) {
            return false;
        }        
        return false;
    }
}
```

### 环形链表II

#### LeetCode 142 Linked List Cycle II

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。使用 O(1) 空间解决此题

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

**输入**：head = [3,2,0,-4], pos = 1
**输出**：返回索引为 1 的链表节点
**解释**：链表中有一个环，其尾部连接到第二个节点。

**提示：**

- 链表中节点的数目范围是 `[0, 10^4]`
- `-10^5 <= Node.val <= 10^5`
- `pos` 为 `-1` 或者链表中的一个 **有效索引** 。

##### 代码实现

先设置fast指针速度为slow的两倍，头结点到入环点距离设为x，入环点到相遇点距离设为y，相遇点回到入环点距离设置为z;
slow走x+y到达相遇点，此时fast多走了n圈，即走了x+y+n(y+z)，由于fast速度比slow快两倍，则有：`2(x+y)=x+y+n(y+z)` 
可化为`x+y=n(y+z)`，这说明头节点到相遇点距离为环长度的整数倍，即，以同一速度分别从头结点和相遇点出发，必定会在相遇点再次相遇。此时返回相遇点的结点。

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast!=null&&fast.next!=null)
        { //找到第一次相遇点
            fast = fast.next.next;
            slow = slow.next;
            if(fast==slow)
            {
                break;
            }
        }
        if(fast==null||fast.next==null){return null;} // 不存在环返回null
        fast = head;        //将一个指针移回头
        while(fast!=slow)   //下次相遇点为入环点
        {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }
}
```



### 相交链表

#### LeetCode 160 相交链表

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。

图示两个链表在节点 c1 开始相交：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

![](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)

**输入**：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
**输出**：Intersected at '8'
**解释**：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

<img src="https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png" style="zoom:80%;" />

**输入**：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
**输出**：null
**解释**：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。

提示：

- listA 中节点数目为 m
- listB 中节点数目为 n
- 0 <= m, n <= 3 * 104
- 1 <= Node.val <= 105
- 0 <= skipA <= m
- 0 <= skipB <= n
- 如果 listA 和 listB 没有交点，intersectVal 为 0
- 如果 listA 和 listB 有交点，intersectVal == listA[skipA + 1] == listB[skipB + 1]

设计一个时间复杂度 O(n) 、仅用 O(1) 内存的解决方案



##### 代码实现

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int i=0,j=0;
        ListNode p1 = headA;
        ListNode p2 = headB;
        for(i=0;p1!=null;i++)
        {
            p1 = p1.next;
        }
        for(j=0;p2!=null;j++)
        {
            p2 = p2.next;
        }
        i=i-j;
        p1 = headA;
        p2 = headB;
        for(;p1!=null&&p2!=null;)
        {
            if(i>0)
            {
                p1 = p1.next;
                i--;
            }
            else if(i<0)
            {
                p2 = p2.next;
                i++;
                
            }
            else{
                break;
            }
            
        }
        for(;p1!=null&&p2!=null;){
            
            if(p1==p2)
            {
                return p1;
            }
            p1 = p1.next;
            p2 = p2.next;
        }
        if(p1==null&&p2==null)
        {
            return null;
        }
        return p1;
    }
}
```

O(m+n) O(1)

另解

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        //变轨次数
        int cnt=0;
        while (p1 != null && p2 != null) {
            if (p1 == p2) return p1;
            p1 = p1.next;
            p2 = p2.next;
            //p1变轨
            if(cnt<2&&p1==null){
                p1=headB;
                cnt++;
            }
            //p2变轨
            if(cnt<2&&p2==null){
                p2=headA;
                cnt++;
            }
        }
        return null;
    }
```



### 19. 删除链表的倒数第N个节点

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**进阶：** 使用一趟扫描实现

![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```bash
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**提示：**

- 链表中结点的数目为 `sz`
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`

##### 代码实现

使用双指针，距离为n，考虑到一个结点的情况，new一个指向头结点的指针（虽然java中不是指针）

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode newhead = new ListNode(0,head);
        ListNode slow = newhead;
        ListNode fast = head;
        for(int i=0;i<n;i++)
        {
            fast=fast.next;
        }
        while(fast!=null)
        {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next=slow.next.next;
        return newhead.next;
    }
}
```



### 206. 反转链表

#### 代码实现

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while(cur!=null)
        {
            ListNode fuck = null;
            fuck = cur.next;
            cur.next = pre;
            pre = cur;
            cur = fuck;
        }
        return pre;
    }
}
```

### 203. 移除链表元素

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

```bash
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
```

#### 代码实现

递归：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if(head==null)
        {
            return head;
        }
        head.next=removeElements(head.next,val);
        return head.val == val ? head.next:head;
    }
}
```

循环：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if(head==null)
        {
            return head;
        }
        ListNode tmp= new ListNode (0,head);
        head = tmp;
        while(tmp!=null&&tmp.next!=null){
           if(tmp.next.val == val){
               tmp.next=tmp.next.next;
               continue;
           }
           tmp = tmp.next;
        }
        
        return head.next;
    }
}
```

### 328. 奇偶链表

#### LeetCode

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

示例 1:

输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
示例 2:

输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL
说明:

应当保持奇数节点和偶数节点的相对顺序。
链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推。

##### 代码实现

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if(head==null||head.next==null||head.next.next==null)
        {
            return head;
        }
        ListNode fu = head;
        ListNode ck = head.next;
        ListNode fuck = ck;
        while (ck!=null&&ck.next!=null)
        {
            fu.next = fu.next.next;
            ck.next = ck.next.next;
            fu = fu.next;
            ck = ck.next;
        }
        if(fu.next!=null)
        {
            fu.next=fu.next.next;
        }
        fu.next=fuck;
        return head;
    }
}
```

### 回文链表

#### LeetCode 234 回文链表

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

##### 代码实现

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public boolean isPalindrome(ListNode head) {
        if(head == null || head.next == null) {
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        ListNode pre = null;
        while(fast!=null&&fast.next!=null){
            fast=fast.next.next;
            ListNode tmp = slow.next;
            slow.next=pre;
            pre = slow;
            slow = tmp;
        }
        if(fast != null) {
            slow = slow.next;
        }
        while(pre!=null&&slow!=null){
            if(pre.val!=slow.val){
                return false;
            }
            pre = pre.next;
            slow = slow.next;
        }
        return true;
    }
   
}
```

### 小结 链表经典问题

> 1. 通过一些测试用例可以节省您的时间。
>
> 使用链表时不易调试。因此，在编写代码之前，自己尝试几个不同的示例来验证您的算法总是很有用的。
>
> 2. 你可以同时使用多个指针。
>
> 有时，当你为链表问题设计算法时，可能需要同时跟踪多个结点。您应该记住需要跟踪哪些结点，并且可以自由地使用几个不同的结点指针来同时跟踪这些结点。
>
> 如果你使用多个指针，最好为它们指定适当的名称，以防将来必须调试或检查代码。
>
> 3. 在许多情况下，你需要跟踪当前结点的前一个结点。
>
> 你无法追溯单链表中的前一个结点。因此，您不仅要存储当前结点，还要存储前一个结点。这在双链表中是不同的，我们将在后面的章节中介绍。
>
> 作者：力扣 (LeetCode)
> 链接：https://leetcode-cn.com/leetbook/read/linked-list/fraqr/
> 来源：力扣（LeetCode）

### 双链表

#### 结点结构

```java
// Definition for doubly-linked list.
class DoublyListNode {
    int val;
    DoublyListNode next, prev;
    DoublyListNode(int x) {val = x;}
}
```

```c++
// Definition for doubly-linked list.
struct DoublyListNode {
    int val;
    DoublyListNode *next, *prev;
    DoublyListNode(int x) : val(x), next(NULL), prev(NULL) {}
};
```

#### 添加操作

如果我们想在现有的结点 `prev` 之后插入一个新的结点 `cur`，我们可以将此过程分为两个步骤：

1. 链接 `cur` 与 `prev` 和 `next`，其中 `next` 是 `prev` 原始的下一个节点；

   <img src="https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/04/28/screen-shot-2018-04-28-at-173045.png" style="zoom:50%;" />

2. 用 `cur` 重新链接 `prev` 和 `next`。

   <img src="https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/04/29/screen-shot-2018-04-28-at-173055.png" style="zoom:67%;" />

#### 删除操作

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/04/18/screen-shot-2018-04-18-at-142428.png)

#### 双链表实现

LeetCode官方源码

```java
public class ListNode {
  int val;
  ListNode next;
  ListNode prev;
  ListNode(int x) { val = x; }
}

class MyLinkedList {
  int size;
  // sentinel nodes as pseudo-head and pseudo-tail
  ListNode head, tail;
  public MyLinkedList() {
    size = 0;
    head = new ListNode(0);
    tail = new ListNode(0);
    head.next = tail;
    tail.prev = head;
  }

  /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
  public int get(int index) {
    // if index is invalid
    if (index < 0 || index >= size) return -1;

    // choose the fastest way: to move from the head
    // or to move from the tail
    ListNode curr = head;
    if (index + 1 < size - index)
      for(int i = 0; i < index + 1; ++i) curr = curr.next;
    else {
      curr = tail;
      for(int i = 0; i < size - index; ++i) curr = curr.prev;
    }

    return curr.val;
  }

  /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
  public void addAtHead(int val) {
    ListNode pred = head, succ = head.next;

    ++size;
    ListNode toAdd = new ListNode(val);
    toAdd.prev = pred;
    toAdd.next = succ;
    pred.next = toAdd;
    succ.prev = toAdd;
  }

  /** Append a node of value val to the last element of the linked list. */
  public void addAtTail(int val) {
    ListNode succ = tail, pred = tail.prev;

    ++size;
    ListNode toAdd = new ListNode(val);
    toAdd.prev = pred;
    toAdd.next = succ;
    pred.next = toAdd;
    succ.prev = toAdd;
  }

  /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
  public void addAtIndex(int index, int val) {
    // If index is greater than the length, 
    // the node will not be inserted.
    if (index > size) return;

    // [so weird] If index is negative, 
    // the node will be inserted at the head of the list.
    if (index < 0) index = 0;

    // find predecessor and successor of the node to be added
    ListNode pred, succ;
    if (index < size - index) {
      pred = head;
      for(int i = 0; i < index; ++i) pred = pred.next;
      succ = pred.next;
    }
    else {
      succ = tail;
      for (int i = 0; i < size - index; ++i) succ = succ.prev;
      pred = succ.prev;
    }

    // insertion itself
    ++size;
    ListNode toAdd = new ListNode(val);
    toAdd.prev = pred;
    toAdd.next = succ;
    pred.next = toAdd;
    succ.prev = toAdd;
  }

  /** Delete the index-th node in the linked list, if the index is valid. */
  public void deleteAtIndex(int index) {
    // if the index is invalid, do nothing
    if (index < 0 || index >= size) return;

    // find predecessor and successor of the node to be deleted
    ListNode pred, succ;
    if (index < size - index) {
      pred = head;
      for(int i = 0; i < index; ++i) pred = pred.next;
      succ = pred.next.next;
    }
    else {
      succ = tail;
      for (int i = 0; i < size - index - 1; ++i) succ = succ.prev;
      pred = succ.prev.prev;
    }

    // delete pred.next 
    --size;
    pred.next = succ;
    succ.prev = pred;
  }
}
```



### 时间复杂度比较

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/04/29/screen-shot-2018-04-28-at-174531.png)

### 合并两个有序链表

#### Leetcode 21 Merge Two Sorted Lists 

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

```bash
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

##### 代码实现

递归写法

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1==null){
            return l2;
        }
        else if(l2==null){
            return l1;
        }
        if(l1.val>=l2.val){
            l2.next = mergeTwoLists(l1,l2.next);
            return l2;
        }
        else{
            l1.next = mergeTwoLists(l1.next,l2);
            return l1;
        }
    }
}
```

### 两数相加

#### Leetcode 2 addTwoNumbers

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)



```bash
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

##### 代码实现

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode out = dummy;
        int sum = 0;
        for(;(l1!=null)||(l2!=null);)
        {
            if(l1!=null)
            {
                sum+=l1.val;
                l1=l1.next;
            }
            if(l2!=null)
            {
                sum+=l2.val;
                l2=l2.next;
            }
            out.next = new ListNode(sum%10);
            sum/=10;
            out=out.next;
        }
        if(sum>0)
        {
            out.next = new ListNode(1);
        }
        return dummy.next;
    }

}
```

###  扁平化多级双向链表

#### LeetCode 430 Flatten a Multilevel Doubly Linked List

多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给你位于列表第一级的头节点，请你扁平化列表，使所有结点出现在单级双链表中。

```baah
输入：head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
输出：[1,2,3,7,8,11,12,9,10,4,5,6]
解释：
```

输入的多级列表如下图所示：

[![eg1](https://raw.githubusercontent.com/Ray-56/image-service/master/picgo/20201027/104455.png)](https://raw.githubusercontent.com/Ray-56/image-service/master/picgo/20201027/104455.png)

扁平化后的链表如下图：

[![eg2](https://raw.githubusercontent.com/Ray-56/image-service/master/picgo/20201027/104604.png)](https://raw.githubusercontent.com/Ray-56/image-service/master/picgo/20201027/104604.png)

示例 2:

```bash
输入：head = [1,2,null,3]
输出：[1,3,2]
解释：

输入的多级列表如下图所示：

  1---2---NULL
  |
  3---NULL
```

示例3：

```bash
输入：head = []
输出：[]
```

**如何表示测试用例中的多级链表？**

以 示例 1 为例：

```bash
 1---2---3---4---5---6--NULL
         |
         7---8---9---10--NULL
             |
             11--12--NULL
```

序列化其中的每一级之后：

```bash
[1,2,3,4,5,6,null]
[7,8,9,10,null]
[11,12,null]
```

为了将每一级都序列化到一起，我们需要每一级中添加值为 null 的元素，以表示没有节点连接到上一级的上级节点。

```bash
[1,2,3,4,5,6,null]
[null,null,7,8,9,10,null]
[null,11,12,null]
```

合并所有序列化结果，并去除末尾的 null 。

```bash
[1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
```

##### 代码实现

dfs遍历

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node prev;
    public Node next;
    public Node child;
};
*/

class Solution {
    private Node fuck = new Node(0);
    public Node flatten(Node head) {
        dfs(head);
        if(head!=null){
            head.prev = null;
        }
        return head;
    }
    private void dfs(Node x){
        if(x==null)
        {
            return;
        }
        Node l = x.child;
        Node r = x.next;
        fuck.next = x;
        x.prev = fuck;
        fuck = x;
        dfs(l);
        x.child = null;
        dfs(r);
    }

}
```

### 复制带随机指针的链表

#### LeetCode 138 Copy List with Random Pointer

给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。

要求返回这个链表的 **深拷贝**。

```text
输入：
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

解释：
节点 1 的值是 1，它的下一个指针和随机指针都指向节点 2 。
节点 2 的值是 2，它的下一个指针指向 null，随机指针指向它自己。
```

**提示：**

1. 你必须返回**给定头的拷贝**作为对克隆列表的引用。

##### 代码实现

时间和空间都是O(N)的算法，维护一个HashMap存结点对应关系；

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/

class Solution {
    public Node copyRandomList(Node head) {
        Node NewNode;
        Node temp = head;
        HashMap<Node, Node> m = new HashMap<>();
        while(temp!=null){
            NewNode = new Node(temp.val);
            m.put(temp,NewNode);
            temp = temp.next;
        }
        temp = head;
        NewNode = m.get(temp);
        while(temp!=null)
        {
            NewNode.next = m.get(temp.next);
            NewNode.random = m.get(temp.random);
            temp = temp.next;
            NewNode = NewNode.next;
        }
        return m.get(head);
    }
}
```

Better solution : 新链表每个结点分别插入在新链表每个结点右边，一次遍历得到一个新旧结点交替的链表，再次遍历根据旧结点random指针，让新结点random指针指向对应节点。再遍历一次逻辑删除旧结点。

时间：O(N)  ；空间： O(1)



### 旋转链表

#### LeetCode 61 Rotate List 

给你一个链表的头节点 `head` ，旋转链表，将链表每个节点向右移动 `k` 个位置。

![](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)

```
输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]
```

##### 代码实现

构成环形链表，在合适的地方断开。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head==null||head.next==null){
            return head;
        }
        ListNode temp = head;
        int counter = 1;
        while(temp.next!=null){
            counter++;
            temp = temp.next;
        }
        if(k==counter){
            return head;
        }

        temp.next = head;
        temp = head;
        
        k%=counter;
        counter -= k;

        counter--;
        while((counter--)>0)
        {
            temp = temp.next;
        }
        head = temp.next;
        temp.next = null;
        return head;
    }
}
```



## Stack

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/06/03/screen-shot-2018-06-02-at-203523.png)

在 LIFO 数据结构中，将`首先处理添加到队列`中的`最新元素`。
与队列不同，栈是一个 LIFO 数据结构。通常，插入操作在栈中被称作入栈 `push` 。与队列类似，总是`在堆栈的末尾添加一个新元素`。但是，删除操作，退栈 `pop` ，将始终`删除`队列中相对于它的`最后一个元素`。

入栈与出栈

![](https://pic.leetcode-cn.com/691e2a8cca120acb18e77379c7cd7eec3835c8c102d1c699303f50accd1e09df-%E5%87%BA%E5%85%A5%E6%A0%88.gif)

栈的实现比队列容易。`动态数组`足以实现堆栈结构。

```java
// "static void main" must be defined in a public class.
class MyStack {
    private List<Integer> data;               // store elements
    public MyStack() {
        data = new ArrayList<>();
    }
    /** Insert an element into the stack. */
    public void push(int x) {
        data.add(x);
    }
    /** Checks whether the queue is empty or not. */
    public boolean isEmpty() {
        return data.isEmpty();
    }
    /** Get the top item from the queue. */
    public int top() {
        return data.get(data.size() - 1);
    }
    /** Delete an element from the queue. Return true if the operation is successful. */
    public boolean pop() {
        if (isEmpty()) {
            return false;
        }
        data.remove(data.size() - 1);
        return true;
    }
};

public class Main {
    public static void main(String[] args) {
        MyStack s = new MyStack();
        s.push(1);
        s.push(2);
        s.push(3);
        for (int i = 0; i < 4; ++i) {
            if (!s.isEmpty()) {
                System.out.println(s.top());
            }
            System.out.println(s.pop());
        }
    }
}
```

使用库函数

```java
// "static void main" must be defined in a public class.
public class Main {
    public static void main(String[] args) {
        // 1. Initialize a stack.
        Stack<Integer> s = new Stack<>();
        // 2. Push new element.
        s.push(5);
        s.push(13);
        s.push(8);
        s.push(6);
        // 3. Check if stack is empty.
        if (s.empty() == true) {
            System.out.println("Stack is empty!");
            return;
        }
        // 4. Pop an element.
        s.pop();
        // 5. Get the top element.
        System.out.println("The top element is: " + s.peek());
        // 6. Get the size of the stack.
        System.out.println("The size is: " + s.size());
    }
}
```



### 最小栈

#### LeetCode 155 Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:

- `MinStack()` initializes the stack object.
- `void push(val)` pushes the element `val` onto the stack.
- `void pop()` removes the element on the top of the stack.
- `int top()` gets the top element of the stack.
- `int getMin()` retrieves the minimum element in the stack.

```
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
```

##### 代码实现

自定义结点，加入min，再来模拟栈

```java
class MinStack {
    private MyNode head;
    /** initialize your data structure here. */
    
    public void push(int val) {
        if(empty()){
            head = new MyNode(val, val, null);
            
        }else{
            head = new MyNode(val,Math.min(val,head.min),head);
        }
    }
    
    public void pop() {
        if(empty()) throw new IllegalStateException("栈为空……");
        head = head.next;
    }
    
    public int top() {
        if (empty()) throw new IllegalStateException("栈为空……");
        return head.val;
    }
    
    public int getMin() {
        if(empty()) throw new IllegalStateException("栈为空……");
        return head.min;
    }

    private boolean empty() {
        return head == null;
    }
}

class MyNode{
    public int val;
    public int min;//最小值
    public MyNode next;

    public MyNode(int val, int min, MyNode next) {
        this.val = val;
        this.min = min;
        this.next = next;
    }

}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```



### 有效的括号

#### LeetCode 20 Valid Parentheses

Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

 

**Example 1:**

```
Input: s = "()"
Output: true
```

**Example 2:**

```
Input: s = "()[]{}"
Output: true
```

**Example 3:**

```
Input: s = "(]"
Output: false
```

**Example 4:**

```
Input: s = "([)]"
Output: false
```

**Example 5:**

```
Input: s = "{[]}"
Output: true
```

 

**Constraints:**

- `1 <= s.length <= 104`
- `s` consists of parentheses only `'()[]{}'`.



##### 代码实现

一次遍历

循环体内：遇到`(`存`)`，遇到`[`存`]`,遇到`{`存`}`, 栈空说明右边括号先与左边括号出现，直接  `return false` ；

取出栈顶，如果当前符号不等于栈顶元素，说明匹配失败；否则可以抵消掉一组括号，继续；

循环结束后：如果栈不为空，说明有只出现左括号未出现右括号与之抵消的，`return false`;

最后`return true`

```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        char a;
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)=='('){
                stack.push(')');
                continue;
            }
            if(s.charAt(i)=='['){
                stack.push(']');
                continue;
            }
            if(s.charAt(i)=='{'){
                stack.push('}');
                continue;
            }
            if(stack.empty()){
                return false;
            }
            a=stack.pop();
            if(s.charAt(i)!=a){
                return false;
            }
            
        }
        if(!stack.empty()){
            return false;
        }
        return true;
    }
}
```

一个小优化，循环结束后可以直接`return stack.isEmpty();`

```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        char a;
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)=='('){
                stack.push(')');
                continue;
            }
            if(s.charAt(i)=='['){
                stack.push(']');
                continue;
            }
            if(s.charAt(i)=='{'){
                stack.push('}');
                continue;
            }
            if(stack.empty()){
                return false;
            }
            a=stack.pop();
            if(s.charAt(i)!=a){
                return false;
            }
            
        }
        return stack.empty();
    }
}
```





### 每日温度

#### LeetCode 739 Daily Temperatures

Given an array of integers `temperatures` represents the daily temperatures, return *an array* `answer` *such that* `answer[i]` *is the number of days you have to wait after the* `ith` *day to get a warmer temperature*. If there is no future day for which this is possible, keep `answer[i] == 0` instead.

 

**Example 1:**

```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

**Example 2:**

```
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
```

**Example 3:**

```
Input: temperatures = [30,60,90]
Output: [1,1,0]
```

 

**Constraints:**

- `1 <= temperatures.length <= 105`
- `30 <= temperatures[i] <= 100`

##### 代码实现

单调栈

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> stack = new Stack<>();
        int[] ret = new int[temperatures.length];
        for(int i=0;i<temperatures.length;i++){
            while(!stack.empty()&&temperatures[i] > temperatures[stack.peek()]){
                int temp = stack.pop();
                ret[temp] = i - temp;
            }
            stack.push(i);
        }
        return ret;
    }
}
```

最优解：倒着来

```java
public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        //从后面开始查找
        for (int i = res.length - 1; i >= 0; i--) {
            int j = i + 1;
            while (j < res.length) {
                if (T[j] > T[i]) {
                    //如果找到就停止while循环
                    res[i] = j - i;
                    break;
                } else if (res[j] == 0) {
                    //如果没找到，并且res[j]==0。说明第j个元素后面没有
                    //比第j个元素大的值，因为这一步是第i个元素大于第j个元素的值，
                    //那么很明显这后面就更没有大于第i个元素的值。直接终止while循环。
                    break;
                } else {
                    //如果没找到，并且res[j]！=0说明第j个元素后面有比第j个元素大的值，
                    //然后我们让j往后挪res[j]个单位，找到那个值，再和第i个元素比较
                    j += res[j];
                }
            }
        }
        return res;
    }
```



### 下一个更大元素 I

#### LeetCode 496 Next Greater Element I

The **next greater element** of some element `x` in an array is the **first greater** element that is **to the right** of `x` in the same array.

You are given two **distinct 0-indexed** integer arrays `nums1` and `nums2`, where `nums1` is a subset of `nums2`.

For each `0 <= i < nums1.length`, find the index `j` such that `nums1[i] == nums2[j]` and determine the **next greater element** of `nums2[j]` in `nums2`. If there is no next greater element, then the answer for this query is `-1`.

Return *an array* `ans` *of length* `nums1.length` *such that* `ans[i]` *is the **next greater element** as described above.*

 

**Example 1:**

```
Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
```

**Example 2:**

```
Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
- 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.
```

 

**Constraints:**

- `1 <= nums1.length <= nums2.length <= 1000`
- `0 <= nums1[i], nums2[i] <= 104`
- All integers in `nums1` and `nums2` are **unique**.
- All the integers of `nums1` also appear in `nums2`.

##### 代码实现

单调栈经典例题

```java
class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer,Integer> res = nextGreater(nums2);
        int[] resores = new int[nums1.length];
        for(int i =0;i<nums1.length;i++){
            resores[i] = res.getOrDefault(nums1[i], -1);;
        }
        return resores;
    }

    private Map<Integer,Integer> nextGreater(int[] arr){
        Stack<Integer> stack = new Stack<>();
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0;i<arr.length;i++){
            while(!stack.isEmpty() &&stack.peek() <arr[i]){
                 map.put(stack.pop(), arr[i]);
            }
            stack.push(arr[i]);
        }
        return map;
    }
}
```



### 逆波兰表达式求值

#### LeetCode 150 Evaluate Reverse Polish Notation

Evaluate the value of an arithmetic expression in [Reverse Polish Notation](http://en.wikipedia.org/wiki/Reverse_Polish_notation).

Valid operators are `+`, `-`, `*`, and `/`. Each operand may be an integer or another expression.

**Note** that division between two integers should truncate toward zero.

It is guaranteed that the given RPN expression is always valid. That means the expression would always evaluate to a result, and there will not be any division by zero operation.

 

**Example 1:**

```
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
```

**Example 2:**

```
Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
```

**Example 3:**

```
Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

 

**Constraints:**

- `1 <= tokens.length <= 104`
- `tokens[i]` is either an operator: `"+"`, `"-"`, `"*"`, or `"/"`, or an integer in the range `[-200, 200]`.

##### 代码实现

```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for(String a:tokens){
            if("+".equals(a)){
                int x = stack.pop();
                int y = stack.pop();
                stack.push(x+y);
            } else if("-".equals(a)){
                int x = stack.pop();
                int y = stack.pop();
                stack.push(y-x);
            } else if("*".equals(a)){
                int x = stack.pop();
                int y = stack.pop();
                stack.push(x*y);
            } else if("/".equals(a)){
                int x = stack.pop();
                int y = stack.pop();
                stack.push(y/x);
            } else{
                stack.push(Integer.parseInt(a));
            }
        }
        return stack.peek();
    }
}
```









## queue

<img src="https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/08/14/screen-shot-2018-05-03-at-151021.png" style="zoom:50%;" />

​    在 FIFO 数据结构中，将`首先处理添加到队列中的第一个元素`。

​    如上图所示，队列是典型的 FIFO 数据结构。插入（insert）操作也称作入队（enqueue），新元素始终被添加在`队列的末尾`。 删除（delete）操作也被称为出队（dequeue)。 你只能移除`第一个元素`。

入队与出队

![](https://pic.leetcode-cn.com/44b3a817f0880f168de9574075b61bd204fdc77748d4e04448603d6956c6428a-%E5%87%BA%E5%85%A5%E9%98%9F.gif)

### 队列实现

```java
// "static void main" must be defined in a public class.

class MyQueue {
    // store elements
    private List<Integer> data;         
    // a pointer to indicate the start position
    private int p_start;            
    public MyQueue() {
        data = new ArrayList<Integer>();
        p_start = 0;
    }
    /** Insert an element into the queue. Return true if the operation is successful. */
    public boolean enQueue(int x) {
        data.add(x);
        return true;
    };    
    /** Delete an element from the queue. Return true if the operation is successful. */
    public boolean deQueue() {
        if (isEmpty() == true) {
            return false;
        }
        p_start++;
        return true;
    }
    /** Get the front item from the queue. */
    public int Front() {
        return data.get(p_start);
    }
    /** Checks whether the queue is empty or not. */
    public boolean isEmpty() {
        return p_start >= data.size();
    }     
};

public class Main {
    public static void main(String[] args) {
        MyQueue q = new MyQueue();
        q.enQueue(5);
        q.enQueue(3);
        if (q.isEmpty() == false) {
            System.out.println(q.Front());
        }
        q.deQueue();
        if (q.isEmpty() == false) {
            System.out.println(q.Front());
        }
        q.deQueue();
        if (q.isEmpty() == false) {
            System.out.println(q.Front());
        }
    }
}
```

```c++
#include <iostream>

class MyQueue {
    private:
        // store elements
        vector<int> data;       
        // a pointer to indicate the start position
        int p_start;            
    public:
        MyQueue() {p_start = 0;}
        /** Insert an element into the queue. Return true if the operation is successful. */
        bool enQueue(int x) {
            data.push_back(x);
            return true;
        }
        /** Delete an element from the queue. Return true if the operation is successful. */
        bool deQueue() {
            if (isEmpty()) {
                return false;
            }
            p_start++;
            return true;
        };
        /** Get the front item from the queue. */
        int Front() {
            return data[p_start];
        };
        /** Checks whether the queue is empty or not. */
        bool isEmpty()  {
            return p_start >= data.size();
        }
};

int main() {
    MyQueue q;
    q.enQueue(5);
    q.enQueue(3);
    if (!q.isEmpty()) {
        cout << q.Front() << endl;
    }
    q.deQueue();
    if (!q.isEmpty()) {
        cout << q.Front() << endl;
    }
    q.deQueue();
    if (!q.isEmpty()) {
        cout << q.Front() << endl;
    }
}
```

**缺点**

上面的实现很简单，但在某些情况下效率很低。 随着起始指针的移动，浪费了越来越多的空间。 当我们有空间限制时，这将是难以接受的。

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/21/screen-shot-2018-07-21-at-153558.png)

让我们考虑一种情况，即我们只能分配一个最大长度为 5 的数组。当我们只添加少于 5 个元素时，我们的解决方案很有效。 例如，如果我们只调用入队函数四次后还想要将元素 10 入队，那么我们可以成功。

但是我们不能接受更多的入队请求，这是合理的，因为现在队列已经满了。但是如果我们将一个元素出队呢？

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/21/screen-shot-2018-07-21-at-153713.png)

实际上，在这种情况下，我们应该能够再接受一个元素。

[![4QHx0S.gif](https://z3.ax1x.com/2021/09/18/4QHx0S.gif)](https://imgtu.com/i/4QHx0S)




### 循环队列

#### LeetCode 622 设计环形队列

设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

你的实现应该支持如下操作：

- `MyCircularQueue(k)`: 构造器，设置队列长度为 k 。
- `Front`: 从队首获取元素。如果队列为空，返回 -1 。
- `Rear`: 获取队尾元素。如果队列为空，返回 -1 。
- `enQueue(value)`: 向循环队列插入一个元素。如果成功插入则返回真。
- `deQueue()`: 从循环队列中删除一个元素。如果成功删除则返回真。
- `isEmpty()`: 检查循环队列是否为空。
- `isFull()`: 检查循环队列是否已满。



```java
MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
circularQueue.enQueue(1);  // 返回 true
circularQueue.enQueue(2);  // 返回 true
circularQueue.enQueue(3);  // 返回 true
circularQueue.enQueue(4);  // 返回 false，队列已满
circularQueue.Rear();  // 返回 3
circularQueue.isFull();  // 返回 true
circularQueue.deQueue();  // 返回 true
circularQueue.enQueue(4);  // 返回 true
circularQueue.Rear();  // 返回 4
```



##### 代码实现

```java
class MyCircularQueue {

    private int[] data;
    private int front, tail;

    public MyCircularQueue(int k) {
        data = new int[k+1];
        front = 0;
        tail = 0;
    }
    
    public boolean enQueue(int value) {
        if(isFull()){
            return false;
        }
        data[tail] = value;
        tail = (tail + 1) % data.length;
        return true;
    }
    
    public boolean deQueue() {
        if(isEmpty()){
            return false;
        }
        front = (front + 1) % data.length;
        return true;
    }
    
    public int Front() {
        if(isEmpty()){
            return -1;
        }
        return data[front];
    }
    
    public int Rear() {
        if(isEmpty()){
            return -1;
        }
        return data[(tail - 1 + data.length) % data.length];
    }
    
    public boolean isEmpty() {
        return front==tail;
    }
    
    public boolean isFull() {
        return (tail + 1) % data.length == front;
    }
}

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * boolean param_1 = obj.enQueue(value);
 * boolean param_2 = obj.deQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * boolean param_5 = obj.isEmpty();
 * boolean param_6 = obj.isFull();
 */
```

官方代码：

```java
class MyCircularQueue {
    
    private int[] data;
    private int head;
    private int tail;
    private int size;

    /** Initialize your data structure here. Set the size of the queue to be k. */
    public MyCircularQueue(int k) {
        data = new int[k];
        head = -1;
        tail = -1;
        size = k;
    }
    
    /** Insert an element into the circular queue. Return true if the operation is successful. */
    public boolean enQueue(int value) {
        if (isFull() == true) {
            return false;
        }
        if (isEmpty() == true) {
            head = 0;
        }
        tail = (tail + 1) % size;
        data[tail] = value;
        return true;
    }
    
    /** Delete an element from the circular queue. Return true if the operation is successful. */
    public boolean deQueue() {
        if (isEmpty() == true) {
            return false;
        }
        if (head == tail) {
            head = -1;
            tail = -1;
            return true;
        }
        head = (head + 1) % size;
        return true;
    }
    
    /** Get the front item from the queue. */
    public int Front() {
        if (isEmpty() == true) {
            return -1;
        }
        return data[head];
    }
    
    /** Get the last item from the queue. */
    public int Rear() {
        if (isEmpty() == true) {
            return -1;
        }
        return data[tail];
    }
    
    /** Checks whether the circular queue is empty or not. */
    public boolean isEmpty() {
        return head == -1;
    }
    
    /** Checks whether the circular queue is full or not. */
    public boolean isFull() {
        return ((tail + 1) % size) == head;
    }
}

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * boolean param_1 = obj.enQueue(value);
 * boolean param_2 = obj.deQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * boolean param_5 = obj.isEmpty();
 * boolean param_6 = obj.isFull();
 */
```

别造轮子了，用库函数就行了：

```java
// "static void main" must be defined in a public class.
public class Main {
    public static void main(String[] args) {
        // 1. Initialize a queue.
        Queue<Integer> q = new LinkedList();
        // 2. Get the first element - return null if queue is empty.
        System.out.println("The first element is: " + q.peek());
        // 3. Push new element.
        q.offer(5);
        q.offer(13);
        q.offer(8);
        q.offer(6);
        // 4. Pop an element.
        q.poll();
        // 5. Get the first element.
        System.out.println("The first element is: " + q.peek());
        // 7. Get the size of the queue.
        System.out.println("The size is: " + q.size());
    }
}
```

```c++
#include <iostream>

int main() {
    // 1. Initialize a queue.
    queue<int> q;
    // 2. Push new element.
    q.push(5);
    q.push(13);
    q.push(8);
    q.push(6);
    // 3. Check if queue is empty.
    if (q.empty()) {
        cout << "Queue is empty!" << endl;
        return 0;
    }
    // 4. Pop an element.
    q.pop();
    // 5. Get the first element.
    cout << "The first element is: " << q.front() << endl;
    // 6. Get the last element.
    cout << "The last element is: " << q.back() << endl;
    // 7. Get the size of the queue.
    cout << "The size is: " << q.size() << endl;
}
```



### 队列&BFS

如何使用 BFS 来找出根结点 `A` 和目标结点 `G` 之间的最短路径:

[![4QD7Gj.gif](https://z3.ax1x.com/2021/09/18/4QD7Gj.gif)](https://imgtu.com/i/4QD7Gj)



> 在特定问题中执行 BFS 之前确定结点和边缘非常重要。通常，结点将是实际结点或是状态，而边缘将是实际边缘或可能的转换。

#### 伪代码模板

```java
/**
 * Return the length of the shortest path between root and target node.
 */
int BFS(Node root, Node target) {
    Queue<Node> queue;  // store all nodes which are waiting to be processed
    int step = 0;       // number of steps neeeded from root to current node
    // initialize
    add root to queue;
    // BFS
    while (queue is not empty) {
        step = step + 1;
        // iterate the nodes which are already in the queue
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            Node cur = the first node in queue;
            return step if cur is target;
            for (Node next : the neighbors of cur) {
                add next to queue;
            }
            remove the first node from queue;
        }
    }
    return -1;          // there is no path from root to target
}
```

1. 如代码所示，在每一轮中，队列中的结点是等待处理的结点。
2. 在每个更外一层的 while 循环之后，我们距离根结点更远一步。变量 step 指示从根结点到我们正在访问的当前结点的距离。

有时，确保我们永远`不会访问一个结点两次`很重要。否则，我们可能陷入无限循环。如果是这样，我们可以在上面的代码中添加一个哈希集来解决这个问题。这是修改后的伪代码：

```java
/**
 * Return the length of the shortest path between root and target node.
 */
int BFS(Node root, Node target) {
    Queue<Node> queue;  // store all nodes which are waiting to be processed
    Set<Node> used;     // store all the used nodes
    int step = 0;       // number of steps neeeded from root to current node
    // initialize
    add root to queue;
    add root to used;
    // BFS
    while (queue is not empty) {
        step = step + 1;
        // iterate the nodes which are already in the queue
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            Node cur = the first node in queue;
            return step if cur is target;
            for (Node next : the neighbors of cur) {
                if (next is not in used) {
                    add next to queue;
                    add next to used;
                }
            }
            remove the first node from queue;
        }
    }
    return -1;          // there is no path from root to target
}
```

> 有两种情况你不需要使用哈希集：
>
> 1. 你完全确定没有循环，例如，在树遍历中；
> 2. 你确实希望多次将结点添加到队列中。

### 岛屿数量

#### LeetCode 200 Number of Islands

给你一个由 `1`（陆地）和 `0`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```markdown
**输入**：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
**输出**：1
```

```markdown
**输入**：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
**输出**：3
```

提示：

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` 的值为 `0` 或 `1`



##### 代码实现

DFS：

```java
class Solution {
    public int numIslands(char[][] grid) {
        int count = 0;
        for(int i=0;i<grid.length;i++){
            for(int j=0;j<grid[0].length;j++){
                if(grid[i][j]=='1'){
                    count++;
                    dfs(grid,i,j);
                }
            }
        }
        return count;
    }
    public void dfs(char[][] grid,int i,int j){
        if(i<0||j<0||i >= grid.length || j >= grid[0].length || grid[i][j] == '0'){
            return;
        }
        grid[i][j] = '0';
        dfs(grid,i+1,j);
        dfs(grid,i,j+1);
        dfs(grid,i-1,j);
        dfs(grid,i,j-1);
    }
}
```

BFS:

```java
class Solution {
    public int numIslands(char[][] grid) {
        int count = 0;
        for(int i=0;i<grid.length;i++){
            for(int j=0;j<grid[0].length;j++){
                if(grid[i][j]=='1'){
                    count++;
                    bfs(grid,i,j);
                }
            }
        }
        return count;
    }
    class MyNode {
        int i, j;

        public MyNode(int i, int j) {
            this.i = i;
            this.j = j;
        }
    } 

    public void bfs(char[][] grid,int i,int j){
        Queue<MyNode> queue = new LinkedBlockingQueue<>();
        queue.offer(new MyNode(i, j));
        while(queue.isEmpty()!=true){
            for(int x=0;x<queue.size();x++){
                MyNode fuck = queue.poll();
                if(fuck.i<0||fuck.j<0||fuck.j>=grid[0].length||fuck.i>=grid.length||grid[fuck.i][fuck.j] == '0'){
                    continue;
                }
                if(grid[fuck.i][fuck.j]=='1'){
                    grid[fuck.i][fuck.j]='0';
                    queue.offer(new MyNode(fuck.i,fuck.j+1));
                    queue.offer(new MyNode(fuck.i+1,fuck.j));
                    queue.offer(new MyNode(fuck.i,fuck.j-1));
                    queue.offer(new MyNode(fuck.i-1,fuck.j));
                }
            }
        }
        
    }
}
```



### 打开转盘锁

#### LeetCode 752 Open the Lock

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

 

示例 1:

```
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
示例 2:
输入: deadends = ["8888"], target = "0009"
输出：1
解释：
把最后一位反向旋转一次即可 "0000" -> "0009"。
示例 3:
输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
输出：-1
解释：
无法旋转到目标数字且不被锁定。
示例 4:
输入: deadends = ["0000"], target = "8888"
输出：-1
```

##### 代码实现

思路，利用队列实现广度优先遍历，找到匹配的字符串，返回深度

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        Set<String> visited = Stream.of(deadends).collect(Collectors.toSet());
        Queue<String> queue = new LinkedList<>();
        queue.offer("0000");
        int step = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            while(size-- >0){
                String cur = queue.poll();
                if (visited.contains(cur)){
                    continue;
                } 
                if(cur.equals(target)){
                    return step;
                }
                visited.add(cur);
                for(int i=0; i<4; i++){
                    char c = cur.charAt(i);
                    String left = cur.substring(0,i) + (c == '0' ? 9 : c-'0'-1) +  cur.substring(i+1);
                    String right = cur.substring(0,i) + (c == '9' ? 0 : c-'0'+1) +  cur.substring(i+1);
                    if(!visited.contains(left)){
                        queue.offer(left);
                    }
                    if(!visited.contains(right)){
                        queue.offer(right);
                    }
                }
            }
            step++;
        }
        return -1;
    }
}
```



### 完全平方数

#### LeetCode 279 Perfect Squares

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

##### 代码实现

思路：BFS

第一次尝试，直接BFS，超时；

第二次尝试，剪枝，扩大了剪枝的范围，大于n不进入循环，大数还是超时；

第三次，写了个visited集，减去数值相等的路径，通过

另外，动态规划也可解

```java
class Solution {
    public int numSquares(int n) {
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        if(n==1){
            return 1;
        }
        queue.offer(0);
        int cnt = 1;
        while(!queue.isEmpty()){
            int size = queue.size();
            int result = 0;
            while(size-- > 0){
                int i = 1;
                int cur = queue.poll();
                result += cur;
                while(result + square(i)<=n){
                    if(result + square(i) == n){
                        return cnt;
                    }
                    if(!visited.contains(result + square(i))){
                        queue.offer(result + square(i));
                        visited.add(result + square(i));
                    }
                    
                    
                    i++;
                }
                result = 0;
            }
            cnt++;
        }
        return cnt;
    }
    private int square(int n){
        return n*n;
    }
}
```

动态规划

```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n+1];
        dp[0]=0;
        for(int i=1;i<=n;i++){
            dp[i] = i;
            for(int j=0;j*j<=i;j++){
                dp[i] = Math.min(dp[i],dp[i-j*j]+1);
            }
        }
        return dp[n];
    }
}
```







## binary-tree









## binary-search-tree











## balanced-search-tree

### 介绍

使用二叉搜索树对某个元素进行查找，虽然平均情况下的时间复杂度是 O(log n)，但是最坏情况下（当所有元素都在树的一侧时）的时间复杂度是 O(n)。因此有了**平衡查找树（Balanced Search Tree）**，平均和最坏情况下的时间复杂度都是 O(log n)

平衡因子（Balance Factor, BF）的概念：左子树高度与右子树高度之差

平衡查找树有很多不同的实现方式：

- AVL 树
- 2-3查找树
- 伸展树
- 红黑树
- B树（也写成B-树，B-tree，中间的“-”代表杠）
- B+ 树

### AVL 树

也叫平衡二叉树（Balanced Binary Tree），AVL是提出这种数据结构的数学家。概念是对于所有结点，BF 的绝对值小于等于1，即**左、右子树的高度之差的绝对值小于等于1**

> 在各种平衡查找树当中，AVL 树和2-3树已经成为了过去，而红黑树（red-black trees）看似变得越来越受人青睐            —— Skiena

AVL 树在实际中并没有太多的用途，可支持 O(log n) 的查找、插入、删除，它比红黑树严格意义上更为平衡，从而导致插入和删除更慢，但遍历却更快。适合用于只需要构建一次，就可以在不重新构造的情况下读取的情况。









## heap

### 大顶堆

- 根结点（亦称为堆顶）的关键字是堆里所有结点关键字中最大者，称为大顶堆。大根堆要求根节点的关键字既大于或等于左子树的关键字值，又大于或等于右子树的关键字值。

- ![image-20211112170804282](https://gitee.com/niimi_sora/pic-upload/raw/master/pics/image-20211112170804282.png)

- 下标运算：$\begin{cases}left=i\times2+1\\right=i\times2+2\\father=(i-1)/2\end{cases}$

- 重要的两个函数：$heapInsert$和$heapify$

- ```java
   private static void heapInsert(int[] arr, int index) {
          while(arr[index]>arr[(index-1)/2]) {
              swap(arr,index,(index-1)/2);
              index = (index-1)/2;
          }
      }
  ```

- ```java
  private static void heapify(int[] arr, int index, int heapSize) {
          int left = index*2+1;
          while (left<heapSize) {
              int largest = (left+1<heapSize && arr[left+1] > arr[left]) ? left+1 : left;
              largest = arr[largest]>arr[index] ? largest:index;
              if(largest==index) break;
              swap(arr, largest, index);
              index = largest;
              left = index*2+1;
          }
      }
  ```

### HeapSort

```java
class Solution {
    public int[] sortArray(int[] nums) {
        if(nums==null||nums.length<2) return nums; 
        for(int i=0;i<nums.length;i++) {  //建立大顶堆
            heapInsert(nums, i);
        }
        int heapSize = nums.length; //大顶堆的length
        swap(nums,0,--heapSize);   //heapSize先减一表示大顶堆的最后一个元素，把最后一个元素和第0个元素互换位置
        while(heapSize>0) {        //每次交换都使heapSize减一了，直到减到0，停止
            heapify(nums, 0, heapSize);  //重新堆化（使改变后的数组变回大顶堆，即找到堆顶）
            swap(nums,0,--heapSize);    //其实就是把最大的元素换到最后去了，并且让堆看不到它
        }//堆空了（看不到任何元素了）说明每次将堆顶拿出来的操作做完了，数组也排好序了
        return nums;
    }
    private static void heapInsert(int[] arr, int index) {
        while(arr[index]>arr[(index-1)/2]) {   //这里要整除，不能写成右移的形式
            swap(arr,index,(index-1)/2);       //如果插入的这个值大于它父亲的值，交换
            index = (index-1)/2;               //坐标也换过去，继续往上窜，直到窜到顶或干不过它上一级了
        }
    }
    private static void heapify(int[] arr, int index, int heapSize) {
        int left = index*2+1;
        while (left<heapSize) {               //左孩子存在，即孩子存在，进入循环
            int largest = (left+1<heapSize && arr[left+1] > arr[left]) ? left+1 : left; //只有右孩子存在且右孩子值大于左孩子值的时候才将右孩子下标赋给largest
            largest = arr[largest]>arr[index] ? largest:index;    //孩子种最大的和当前选定的比较
            if(largest==index) break;                             //说明当前这个值更大，干过它孩子了，停止
            swap(arr, largest, index);         //到这儿说明当前这个值没干过它孩子种最大的那个，和那个值互换位置
            index = largest;    //注意：第一次写漏了，不只是要把left换成原来largest的left,自己的index也得换过去
            left = index*2+1;   //把left换成当前新选定位置（值是原来的值，但是它位置换到它孩子那儿去了）的左孩子下标
        }
    }
    private static void swap(int[] arr, int i, int j) {
        if(i!=j) {             //避免同地址异或给置零了
            arr[i] = arr[i]^arr[j];  //只要地址不同就能换，值是可以相同的
            arr[j] = arr[i]^arr[j];
            arr[i] = arr[i]^arr[j];
        }
    }
}
```















## trie









## union-find









## hash-table

[见Java基础部分](./Java基础/#hashmap)

